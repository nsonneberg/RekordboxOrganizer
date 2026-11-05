"""Tests for metadata.py, with fake HTTP responses (no network)."""

import os
import shutil
import tempfile
import unittest
import urllib.parse

import metadata
import organizer
from metadata import MetadataResult, ProviderError, Query


def track(tid, name, artist="", length=0):
    return organizer.Track(tid, name, artist, "", "", 0, "", 0, 0, "", "", 0, "", length=length)


class FakeWeb:
    """Answers URLs by substring; records what was asked."""

    def __init__(self, routes, fail=None):
        self.routes = routes
        self.fail = fail
        self.asked = []

    def __call__(self, url, headers):
        self.asked.append(url)
        self.headers = headers
        if self.fail:
            raise ProviderError(self.fail)
        for key, value in self.routes:
            if key in urllib.parse.unquote(url):
                if isinstance(value, Exception):
                    raise value
                return value
        return {}


def mb_recording(rid, title, artist, length_ms, tags=()):
    return {"id": rid, "title": title, "length": length_ms,
            "artist-credit": [{"name": artist, "joinphrase": ""}],
            "tags": [{"name": t, "count": 1} for t in tags]}


class ParseTests(unittest.TestCase):
    def test_artist_prefix_and_remix(self):
        q = metadata.parse("Orla Finch - Glass Harbor (Tomas Reed Remix)", "", 240)
        self.assertEqual((q.artist, q.title, q.version, q.length),
                         ("Orla Finch", "Glass Harbor", "Tomas Reed Remix", 240.0))

    def test_original_and_junk_are_not_versions(self):
        for name in ("Song (Original Mix)", "Song (Extended Mix) [FREE DL]", "Song (Radio Edit)"):
            q = metadata.parse(name, "Someone")
            self.assertEqual((q.title, q.version), ("Song", ""), name)

    def test_bootlegs_and_flips_are_versions(self):
        self.assertEqual(metadata.parse("Paper Lanterns (Zeno Vale Bootleg)", "Mira Solen").version,
                         "Zeno Vale Bootleg")
        self.assertEqual(metadata.parse("NEON PICNIC! (kovi flip)").version, "kovi flip")

    def test_artist_repeated_in_title(self):
        q = metadata.parse("Dana Quill - Quiet Hours", "Dana Quill")
        self.assertEqual(q.title, "Quiet Hours")


class MatchTests(unittest.TestCase):
    q = Query("Orla Finch", "Glass Harbor", "Tomas Reed Remix", 240)

    def test_same_remix_matches(self):
        self.assertGreater(metadata.match(self.q, "Orla Finch", "Glass Harbor (Tomas Reed Remix)", 242), 0.8)

    def test_original_does_not_match_a_remix(self):
        self.assertEqual(metadata.match(self.q, "Orla Finch", "Glass Harbor", 228), 0)
        plain = Query("Orla Finch", "Glass Harbor", "", 228)
        self.assertEqual(metadata.match(plain, "Orla Finch", "Glass Harbor (Tomas Reed Remix)", 228), 0)

    def test_someone_elses_remix_does_not_match(self):
        self.assertEqual(metadata.match(self.q, "Orla Finch", "Glass Harbor (Lena Hart Remix)", 240), 0)

    def test_wrong_artist_or_length(self):
        self.assertEqual(metadata.match(self.q, "Rook Avery", "Glass Harbor (Tomas Reed Remix)", 240), 0)
        self.assertEqual(metadata.match(self.q, "Orla Finch", "Glass Harbor (Tomas Reed Remix)", 400), 0)

    def test_ambiguous_title_without_artist_is_never_top_quality(self):
        q = Query("", "Tangerine", "", 0)
        self.assertLess(metadata.match(q, "Anyone", "Tangerine"), 0.6)

    def test_equally_good_matches_that_disagree_are_ambiguous(self):
        a = (0.9, [("Techno", 1)], "a")
        b = (0.88, [("Pop", 1)], "b")
        self.assertIsNone(metadata.best_match(self.q, [a, b]))
        c = (0.88, [("Techno", 1), ("House", 1)], "c")
        self.assertEqual(metadata.best_match(self.q, [a, c])[2], "a")     # they agree
        self.assertEqual(metadata.best_match(self.q, [a, (0.5, [("Pop", 1)], "d")])[2], "a")


class MusicBrainzTests(unittest.TestCase):
    def provider(self, routes, **kw):
        web = FakeWeb(routes, **kw)
        return metadata.MusicBrainz(fetch=web, sleep=lambda s: None, version="test"), web

    def test_recording_genres(self):
        rec = mb_recording("r1", "Quiet Hours", "Dana Quill", 360000)
        p, web = self.provider([("/recording?", {"recordings": [rec]}),
                                ("/recording/r1", {"genres": [{"name": "techno", "count": 3}]})])
        r = p.lookup(Query("Dana Quill", "Quiet Hours", "", 361))
        self.assertEqual((r.source, r.genres, r.level), ("MusicBrainz", [("techno", 3)], "recording"))
        self.assertGreater(r.quality, 0.8)
        self.assertIn("RekordboxOrganizer/test (", web.headers["User-Agent"])

    def test_release_group_fallback_is_lower_quality_and_skips_compilations(self):
        rec = mb_recording("r1", "Song", "Artist", 200000)
        detail = {"genres": [], "tags": [], "releases": [
            {"release-group": {"id": "g1"}}, {"release-group": {"id": "g2"}}]}
        p, _ = self.provider([("/recording?", {"recordings": [rec]}), ("/recording/r1", detail),
                              ("/release-group/g1", {"primary-type": "Compilation",
                                                     "genres": [{"name": "pop", "count": 9}]}),
                              ("/release-group/g2", {"primary-type": "Single", "title": "Song EP",
                                                     "genres": [{"name": "house", "count": 2}]})])
        r = p.lookup(Query("Artist", "Song", "", 200))
        self.assertEqual((r.genres, r.level), ([("house", 2)], "release-group"))
        self.assertLess(r.quality, 0.75)

    def test_no_genres_or_no_match(self):
        rec = mb_recording("r1", "Song", "Artist", 200000)
        p, _ = self.provider([("/recording?", {"recordings": [rec]}),
                              ("/recording/r1", {"genres": [], "tags": [], "releases": []})])
        self.assertIsNone(p.lookup(Query("Artist", "Song", "", 200)))
        p, _ = self.provider([("/recording?", {"recordings": [
            mb_recording("r2", "Other Song", "Artist", 200000)]})])
        self.assertIsNone(p.lookup(Query("Artist", "Song", "", 200)))

    def test_api_failure_raises_and_disables(self):
        p, web = self.provider([], fail="HTTP 500")
        for _ in range(metadata.MAX_FAILURES):
            with self.assertRaises(ProviderError):
                p.lookup(Query("A", "Song", "", 0))
        self.assertTrue(p.disabled)
        asked = len(web.asked)
        with self.assertRaises(ProviderError):
            p.lookup(Query("A", "Song", "", 0))
        self.assertEqual(len(web.asked), asked)          # no more requests

    def test_slow_down_is_retried(self):
        calls = []

        def flaky(url, headers):
            calls.append(url)
            if len(calls) == 1:
                raise ProviderError("HTTP 503")
            return {"recordings": []}
        p = metadata.MusicBrainz(fetch=flaky, sleep=lambda s: None)
        self.assertIsNone(p.lookup(Query("A", "Song", "", 0)))
        self.assertEqual(len(calls), 2)
        self.assertEqual(p.failures, 0)


class DiscogsTests(unittest.TestCase):
    def test_tracklist_match_and_various_skipped(self):
        search = {"results": [{"id": 1, "title": "Various - Club Hits"},
                              {"id": 2, "title": "Dana Quill - Quiet Hours EP"}]}
        release = {"id": 2, "title": "Quiet Hours EP", "year": 2019,
                   "artists": [{"name": "Dana Quill"}], "styles": ["Techno"],
                   "genres": ["Electronic"],
                   "tracklist": [{"title": "Quiet Hours", "duration": "6:01"},
                                 {"title": "Hypnotized", "duration": "5:40"}]}
        web = FakeWeb([("/database/search", search), ("/releases/2", release)])
        p = metadata.Discogs(token="abc", fetch=web, sleep=lambda s: None)
        r = p.lookup(Query("Dana Quill", "Quiet Hours", "", 361))
        self.assertEqual(r.genres, [("Techno", 1.0), ("Electronic", 0.3)])
        self.assertEqual(r.level, "release")
        self.assertNotIn("releases/1", " ".join(web.asked))
        self.assertEqual(web.headers["Authorization"], "Discogs token=abc")
        self.assertIn("Quiet Hours EP, 2019", r.describe)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.cache = os.path.join(self.dir, "meta.json")
        self.rec = mb_recording("r1", "Quiet Hours", "Dana Quill", 360000)

    def provider(self, fail=None):
        self.web = FakeWeb([("/recording?", {"recordings": [self.rec]}),
                            ("/recording/r1", {"genres": [{"name": "techno", "count": 1}]})],
                           fail=fail)
        return metadata.MusicBrainz(fetch=self.web, sleep=lambda s: None)

    def run_lookup(self, provider, now):
        tracks = [track("1", "Dana Quill - Quiet Hours", length=360),
                  track("2", "Nobody - Unknown Song", length=200)]
        return metadata.lookup_tracks(tracks, [provider], lambda m: None, cache_file=self.cache,
                                      now=now)

    def test_hits_misses_expiry_and_version(self):
        out, stats = self.run_lookup(self.provider(), now=1_000_000)
        self.assertEqual(list(out), ["1"])
        self.assertEqual((stats.matched, stats.not_found, stats.failed), (1, 1, 0))
        # Again: everything from the cache, no requests.
        p = self.provider()
        out, stats = self.run_lookup(p, now=1_000_000 + 86400)
        self.assertEqual((list(out), stats.from_cache, len(self.web.asked)), (["1"], 2, 0))
        # The "not found" expires sooner than the match.
        p = self.provider()
        self.run_lookup(p, now=1_000_000 + (metadata.MISS_DAYS + 1) * 86400)
        self.assertTrue(self.web.asked)
        self.assertTrue(all("Unknown" in u for u in self.web.asked))
        # A new cache version throws everything away.
        with open(self.cache, encoding="utf-8") as f:
            text = f.read().replace(f'"version": {metadata.CACHE_VERSION}', '"version": 0')
        with open(self.cache, "w", encoding="utf-8") as f:
            f.write(text)
        self.assertEqual(metadata.load_cache(self.cache), {})

    def test_offline_is_not_cached_and_stops_quickly(self):
        tracks = [track(str(i), f"Artist - Song {i}") for i in range(10)]
        p = self.provider(fail="no internet")
        out, stats = metadata.lookup_tracks(tracks, [p], lambda m: None, cache_file=self.cache)
        self.assertEqual((out, stats.failed), ({}, 10))
        self.assertEqual(len(self.web.asked), metadata.MAX_FAILURES)   # then skipped
        self.assertFalse(os.path.exists(self.cache))                    # nothing cached

    def test_corrupt_cache_is_ignored(self):
        with open(self.cache, "w") as f:
            f.write("{oops")
        out, _ = self.run_lookup(self.provider(), now=5)
        self.assertEqual(list(out), ["1"])


class MergeTests(unittest.TestCase):
    def test_merge_weights_by_quality(self):
        a = MetadataResult("MusicBrainz", [("techno", 2)], 0.9, "recording", "a")
        b = MetadataResult("Discogs", [("Techno", 1.0), ("Electronic", 0.3)], 0.6, "release", "b")
        m = metadata.merge([a, b, None])
        self.assertEqual(m.source, "MusicBrainz + Discogs")
        self.assertEqual(m.quality, 0.9)
        self.assertIsNone(metadata.merge([None]))


if __name__ == "__main__":
    unittest.main()
