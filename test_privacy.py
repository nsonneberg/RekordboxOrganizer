"""Token storage, settings migration, online-lookup consent and metadata retention."""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import credentials
import metadata
import organizer
from test_metadata import FakeWeb, mb_recording, track


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, name):
        return self.store.get((service, name))

    def set_password(self, service, name, value):
        self.store[(service, name)] = value

    def delete_password(self, service, name):
        self.store.pop((service, name), None)


class CredentialTests(unittest.TestCase):
    def test_round_trip_and_delete(self):
        kr = FakeKeyring()
        with mock.patch.object(credentials, "_keyring", lambda: kr):
            self.assertTrue(credentials.set_secret("t", "abc"))
            self.assertEqual(credentials.get_secret("t"), "abc")
            self.assertTrue(credentials.set_secret("t", ""))
            self.assertEqual(credentials.get_secret("t"), "")

    def test_no_store(self):
        with mock.patch.object(credentials, "_keyring", lambda: None):
            self.assertFalse(credentials.available())
            self.assertFalse(credentials.set_secret("t", "abc"))
            self.assertEqual(credentials.get_secret("t"), "")


class AppSettingsTests(unittest.TestCase):
    """The real window, with settings.json and the credential store faked."""

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
        self.kr = FakeKeyring()

    def write(self, data):
        with open(self.settings, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def read(self):
        with open(self.settings, encoding="utf-8") as f:
            return json.load(f)

    def make(self, keyring=True):
        p = mock.patch.object(credentials, "_keyring", lambda: self.kr if keyring else None)
        p.start()
        self.addCleanup(p.stop)
        return self.app_mod.App(self.root)

    def test_plain_text_token_moves_to_the_credential_store(self):
        self.write({"version": 2, "values": {"discogs_token": "secret123", "bpm_step": 4}})
        a = self.make()
        self.assertEqual(a.values["discogs_token"].get(), "secret123")
        self.assertEqual(self.kr.store[("RekordboxOrganizer", "discogs_token")], "secret123")
        self.assertNotIn("secret123", open(self.settings, encoding="utf-8").read())
        self.assertEqual(self.read()["values"]["bpm_step"], 4)        # other settings kept
        a.save_settings()
        self.assertNotIn("secret123", open(self.settings, encoding="utf-8").read())

    def test_without_a_store_an_old_token_is_kept_not_lost(self):
        self.write({"version": 2, "values": {"discogs_token": "secret123"}})
        a = self.make(keyring=False)
        self.assertEqual(a.values["discogs_token"].get(), "secret123")
        a.save_settings()
        self.assertEqual(self.read()["values"]["discogs_token"], "secret123")

    def test_a_new_token_is_never_written_to_settings(self):
        a = self.make(keyring=False)
        a.values["discogs_token"].set("typed-in")
        a.save_settings()
        self.assertNotIn("typed-in", open(self.settings, encoding="utf-8").read())
        b = self.make()
        b.values["discogs_token"].set("typed-in")
        b.save_settings()
        self.assertNotIn("typed-in", open(self.settings, encoding="utf-8").read())
        self.assertEqual(self.kr.store[("RekordboxOrganizer", "discogs_token")], "typed-in")

    def test_discogs_needs_its_own_consent(self):
        self.write({"version": 2, "metadata_consent": "2026-01-01",
                    "options": {"genre_metadata": True, "discogs_lookup": True}})
        a = self.make()
        self.assertTrue(a.vars["genre_metadata"].get())
        self.assertFalse(a.vars["discogs_lookup"].get())    # MusicBrainz consent isn't enough
        with mock.patch.object(self.app_mod.messagebox, "askyesno", return_value=False) as ask:
            a.vars["discogs_lookup"].set(True)
        self.assertIn("Discogs", ask.call_args[0][1])
        self.assertIn("token", ask.call_args[0][1])
        self.assertFalse(a.vars["discogs_lookup"].get())
        with mock.patch.object(self.app_mod.messagebox, "askyesno", return_value=True):
            a.vars["discogs_lookup"].set(True)
        a.save_settings()
        self.assertTrue(self.read()["discogs_consent"])

    def test_token_kept_when_the_store_doesnt_really_keep_it(self):
        class Forgetful(FakeKeyring):          # accepts the token, then can't return it
            def set_password(self, service, name, value):
                pass
        self.kr = Forgetful()
        self.write({"version": 2, "values": {"discogs_token": "secret123", "bpm_step": 4}})
        a = self.make()
        self.assertEqual(a.values["discogs_token"].get(), "secret123")
        self.assertEqual(self.read()["values"]["discogs_token"], "secret123")   # not lost
        a.save_settings()
        self.assertEqual(self.read()["values"]["discogs_token"], "secret123")

    def test_failed_settings_write_keeps_the_old_file(self):
        self.write({"version": 2, "values": {"discogs_token": "secret123", "bpm_step": 4}})
        before = open(self.settings, encoding="utf-8").read()

        def broken_dump(data, f, **kw):
            f.write("{half")
            raise OSError("disk full")
        with mock.patch.object(self.app_mod.json, "dump", broken_dump):
            a = self.make()                    # migration can't rewrite the file...
        self.assertEqual(a.values["discogs_token"].get(), "secret123")
        self.assertEqual(open(self.settings, encoding="utf-8").read(), before)   # ...intact
        self.assertFalse(os.path.exists(self.settings + ".partial"))

    def test_online_lookup_needs_explicit_consent(self):
        self.write({"version": 2, "options": {"genre_metadata": True, "discogs_lookup": True}})
        a = self.make()
        self.assertFalse(a.vars["genre_metadata"].get())   # saved without consent: off
        with mock.patch.object(self.app_mod.messagebox, "askyesno", return_value=False) as ask:
            a.vars["genre_metadata"].set(True)
            self.assertTrue(ask.called)
            self.assertIn("artist and title", ask.call_args[0][1])
            self.assertIn("MusicBrainz", ask.call_args[0][1])
            self.assertNotIn("Discogs", ask.call_args[0][1])
        self.assertFalse(a.vars["genre_metadata"].get())
        with mock.patch.object(self.app_mod.messagebox, "askyesno", return_value=True):
            a.vars["genre_metadata"].set(True)
        self.assertTrue(a.vars["genre_metadata"].get())
        a.save_settings()
        data = self.read()
        self.assertTrue(data["metadata_consent"])
        self.assertTrue(data["options"]["genre_metadata"])
        with mock.patch.object(self.app_mod.messagebox, "askyesno") as ask:
            b = self.make()
            self.assertTrue(b.vars["genre_metadata"].get())   # consent remembered
            self.assertFalse(ask.called)


class ProviderChoiceTests(unittest.TestCase):
    def providers(self, **opt):
        seen = {}

        def fake(tracks, providers, log, progress=None):
            seen["names"] = [p.name for p in providers]
            return {}, metadata.LookupStats()
        with mock.patch.object(metadata, "lookup_tracks", fake):
            organizer.metadata_genres([track("1", "A - B")], organizer.Options(**opt),
                                      lambda m: None)
        return seen["names"]

    def test_discogs_is_a_separate_opt_in(self):
        self.assertEqual(self.providers(genre_metadata=True), ["MusicBrainz"])
        self.assertEqual(self.providers(genre_metadata=True, discogs_lookup=True),
                         ["MusicBrainz", "Discogs"])

    def test_metadata_off_by_default(self):
        opt = organizer.Options()
        self.assertFalse(opt.genre_metadata)
        self.assertFalse(opt.discogs_lookup)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.cache = os.path.join(self.dir, "meta.json")
        search = {"results": [{"id": 2, "title": "Dana Quill - Quiet Hours EP"}]}
        release = {"id": 2, "title": "Quiet Hours EP", "artists": [{"name": "Dana Quill"}],
                   "styles": ["Techno"], "genres": ["Electronic"],
                   "tracklist": [{"title": "Quiet Hours", "duration": "6:01"}]}
        self.dweb = FakeWeb([("/database/search", search), ("/releases/2", release)])
        rec = mb_recording("r1", "Quiet Hours", "Dana Quill", 360000)
        self.mweb = FakeWeb([("/recording?", {"recordings": [rec]}),
                             ("/recording/r1", {"genres": [{"name": "techno", "count": 1}]})])

    def lookup(self, now, offline=False):
        providers = [metadata.MusicBrainz(fetch=self.mweb, sleep=lambda s: None),
                     metadata.Discogs(fetch=self.dweb, sleep=lambda s: None)]
        return metadata.lookup_tracks([track("1", "Dana Quill - Quiet Hours", length=361)],
                                      providers, lambda m: None, cache_file=self.cache, now=now,
                                      offline=offline)

    def test_discogs_answers_expire_after_six_hours(self):
        out, _ = self.lookup(1_000_000)
        self.assertEqual(out["1"].source, "MusicBrainz + Discogs")
        self.assertEqual([p.source for p in out["1"].parts], ["MusicBrainz", "Discogs"])
        self.mweb.asked.clear()
        self.dweb.asked.clear()
        self.lookup(1_000_000 + 5 * 3600)
        self.assertEqual((self.mweb.asked, self.dweb.asked), ([], []))
        self.lookup(1_000_000 + 7 * 3600)
        self.assertEqual(self.mweb.asked, [])          # MusicBrainz still cached
        self.assertTrue(self.dweb.asked)               # Discogs asked again

    def test_expired_discogs_answers_are_deleted_even_offline(self):
        self.lookup(1_000_000)
        asked = (len(self.mweb.asked), len(self.dweb.asked))
        out, stats = self.lookup(1_000_000 + 7 * 3600, offline=True)
        self.assertEqual((len(self.mweb.asked), len(self.dweb.asked)), asked)   # nothing sent
        with open(self.cache, encoding="utf-8") as f:
            entries = json.load(f)["entries"]
        self.assertFalse(any(k.startswith("Discogs|") for k in entries))
        self.assertEqual(out["1"].source, "MusicBrainz")

    def test_offline_sends_nothing(self):
        out, stats = self.lookup(1_000_000, offline=True)
        self.assertEqual((out, self.mweb.asked, self.dweb.asked), ({}, [], []))
        self.assertEqual(stats.not_looked_up, 1)
        self.assertFalse(os.path.exists(self.cache))

    def test_cache_holds_no_merged_parts(self):
        self.lookup(1_000_000)
        with open(self.cache, encoding="utf-8") as f:
            self.assertNotIn('"parts"', f.read())


class PacingTests(unittest.TestCase):
    def test_requests_are_spaced(self):
        waits = []
        p = metadata.MusicBrainz(fetch=lambda url, headers: {}, sleep=waits.append)
        p.get("https://x/1")
        p.get("https://x/2")
        self.assertEqual(len(waits), 1)
        self.assertGreater(waits[0], 0.9)
        self.assertLessEqual(waits[0], metadata.MusicBrainz.interval)

    def test_discogs_pacing_stays_under_its_limits(self):
        self.assertGreaterEqual(metadata.Discogs().interval, 60 / 25)      # 25 a minute
        self.assertGreaterEqual(metadata.Discogs(token="x").interval, 60 / 60)

    def test_user_agent_has_no_email(self):
        p = metadata.MusicBrainz(version="1.2.3")
        ua = p.headers["User-Agent"]
        self.assertIn("RekordboxOrganizer/1.2.3", ua)
        self.assertIn("https://github.com/", ua)
        self.assertNotIn("@", ua)


if __name__ == "__main__":
    unittest.main()
