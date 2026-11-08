"""The Update now path: real HTTP on a local server for redirects and failures. No library, no GitHub."""

import hashlib
import http.server
import io
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from unittest import mock

import app
import updater

REAL_RUN = subprocess.run
PAYLOAD = b"the new version " * 2000
OK = (200, {}, PAYLOAD)


class Server:
    """Answers each path from a script of responses; the last one repeats."""

    def __init__(self):
        self.script, self.requests = {}, []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.requests.append((self.path, self.headers.get("Accept")))
                queue = outer.script[self.path]
                step = queue.pop(0) if len(queue) > 1 else queue[0]
                if step == "drop":                  # connection lost before any answer
                    self.close_connection = True
                    return
                status, headers, body = step
                headers = dict(headers)
                length = headers.pop("Declared-Length", len(body))
                self.send_response(status)
                self.send_header("Content-Length", str(length))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                if length != len(body):             # connection lost partway through
                    self.close_connection = True

        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def close(self):
        self.http.shutdown()
        self.http.server_close()


def info_for(url, payload=PAYLOAD):
    return {"url": url, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class DownloadCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rbo-update-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.server = Server()
        self.addCleanup(self.server.close)
        self.dest = os.path.join(self.dir, "app.new")
        self.progress, self.messages = [], []
        base = self.server.base
        sleeping = mock.patch.object(updater.time, "sleep")
        self.sleep = sleeping.start()
        self.addCleanup(sleeping.stop)
        for p in (mock.patch.object(updater, "_is_trusted", lambda url: url.startswith(base)),
                  mock.patch.object(updater, "_context", lambda: None)):
            p.start()
            self.addCleanup(p.stop)

    def serve(self, *steps, path="/file"):
        self.server.script[path] = list(steps)
        return self.server.base + path

    def download(self, url, info=None):
        updater._download(info or info_for(url), self.dest,
                          lambda d, t: self.progress.append((d, t)), self.messages.append)

    def assert_nothing_kept(self):
        self.assertEqual(os.listdir(self.dir), [])

    def assert_complete(self):
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), PAYLOAD)
        self.assertEqual(os.listdir(self.dir), ["app.new"])
        self.assertEqual(self.progress[-1], (len(PAYLOAD), len(PAYLOAD)))


class RedirectTests(DownloadCase):
    def test_follows_github_redirect_to_storage_asking_for_the_file(self):
        target = self.serve(OK, path="/storage")
        start = self.serve((302, {"Location": target}, b""))
        self.download(start)
        self.assert_complete()
        self.assertEqual([r[1] for r in self.server.requests], ["application/octet-stream"] * 2)

    def test_redirect_away_from_github_is_refused(self):
        start = self.serve((302, {"Location": "http://elsewhere.invalid/x"}, b""))
        with self.assertRaisesRegex(updater.UpdateError, "unexpected"):
            self.download(start)
        self.assert_nothing_kept()
        self.assertEqual(len(self.server.requests), 1)
        self.sleep.assert_not_called()


class TrustTests(unittest.TestCase):
    def test_only_github_over_https(self):
        good = ["https://github.com/a/b", "https://api.github.com/x",
                "https://release-assets.githubusercontent.com/x", "https://objects.githubusercontent.com/x"]
        bad = ["http://github.com/a", "https://github.com.evil.example/a",
               "https://evilgithubusercontent.com/a", "https://example.com/github.com",
               "https://github.com:8443/a", "https://user@github.com/a", "https://github.com:abc/a",
               "https://[bad/a", "https://evil.example\\@github.com/a", "ftp://github.com/a"]
        self.assertTrue(all(map(updater._is_trusted, good)))
        self.assertFalse(any(map(updater._is_trusted, bad)))


class RetryTests(DownloadCase):
    GATEWAY = (504, {}, b"Gateway Time-out")

    def test_a_504_is_retried_and_the_user_is_told(self):
        url = self.serve(self.GATEWAY, OK)
        self.download(url)
        self.assert_complete()
        self.assertEqual(len(self.server.requests), 2)
        self.sleep.assert_called_once_with(2)
        self.assertRegex(self.messages[0], r"HTTP 504.*try 2 of 4")

    def test_a_504_that_persists_stops_after_four_tries_and_says_so(self):
        url = self.serve(self.GATEWAY)
        with self.assertRaisesRegex(updater.UpdateError,
                                    r"after 4 tries \(HTTP 504 .*127\.0\.0\.1\).*wasn't changed"):
            self.download(url)
        self.assertEqual(len(self.server.requests), 4)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [2, 5, 15])
        self.assert_nothing_kept()

    def test_a_504_on_the_download_link_switches_to_the_api_link(self):
        url = self.serve(self.GATEWAY)
        api = self.serve(OK, path="/api-asset")
        self.download(url, dict(info_for(url), api_url=api))
        self.assert_complete()
        self.assertEqual(self.server.requests, [("/file", "application/octet-stream"),
                                                ("/api-asset", "application/octet-stream")])

    def test_both_links_failing_still_stops_after_four_tries(self):
        url = self.serve(self.GATEWAY)
        api = self.serve(self.GATEWAY, path="/api-asset")
        with self.assertRaisesRegex(updater.UpdateError, "after 4 tries.*HTTP 504"):
            self.download(url, dict(info_for(url), api_url=api))
        self.assertEqual([r[0] for r in self.server.requests], ["/file", "/api-asset"] * 2)
        self.assert_nothing_kept()

    def test_a_404_is_not_retried(self):
        url = self.serve((404, {}, b"no"))
        with self.assertRaisesRegex(updater.UpdateError, "HTTP 404"):
            self.download(url)
        self.assertEqual(len(self.server.requests), 1)
        self.sleep.assert_not_called()

    def test_retry_after_is_honoured_up_to_a_limit(self):
        url = self.serve((503, {"Retry-After": "9"}, b""), (503, {"Retry-After": "9999"}, b""), OK)
        self.download(url)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [9, updater.MAX_RETRY_AFTER])

    def test_a_dropped_connection_is_retried(self):
        url = self.serve("drop", OK)
        self.download(url)
        self.assert_complete()

    def test_the_release_check_retries_then_succeeds(self):
        body = json.dumps({"tag_name": "v9.0.0", "assets": []}).encode()
        api = self.serve(self.GATEWAY, (200, {}, body), path="/api")
        with mock.patch.object(updater, "LATEST_API", api):
            found = updater.check()
        self.assertEqual(found["version"], "9.0.0")
        self.assertIsNone(found["url"])

    def test_the_release_check_gives_up(self):
        api = self.serve(self.GATEWAY, path="/api")
        with mock.patch.object(updater, "LATEST_API", api):
            with self.assertRaisesRegex(updater.UpdateError, "after 3 tries"):
                updater.check()
        self.assertEqual(len(self.server.requests), 3)


class PartialDownloadTests(DownloadCase):
    CUT = (200, {"Declared-Length": len(PAYLOAD)}, PAYLOAD[:1000])

    def test_a_cut_off_download_is_retried_from_the_start(self):
        url = self.serve(self.CUT, OK)
        self.download(url)
        self.assert_complete()
        self.assertEqual(len(self.server.requests), 2)
        self.assertIn((0, len(PAYLOAD)), self.progress[1:])

    def test_a_download_that_stays_cut_off_is_never_kept(self):
        url = self.serve(self.CUT)
        with self.assertRaisesRegex(updater.UpdateError, "after 4 tries.*1,000 of"):
            self.download(url)
        self.assertEqual(len(self.server.requests), 4)
        self.assert_nothing_kept()

    def test_a_complete_response_of_the_wrong_size_is_not_kept(self):
        url = self.serve((200, {}, PAYLOAD[:-1]))
        with self.assertRaisesRegex(updater.UpdateError, "bytes"):
            self.download(url)
        self.assert_nothing_kept()

    def test_a_disk_problem_is_not_retried(self):
        url = self.serve(OK)
        self.dest = os.path.join(self.dir, "missing-folder", "app.new")
        with self.assertRaises(FileNotFoundError):
            self.download(url)
        self.assertEqual(len(self.server.requests), 1)


class IntegrityTests(DownloadCase):
    def test_a_wrong_checksum_is_thrown_away_without_retrying(self):
        url = self.serve(OK)
        info = dict(info_for(url), sha256="0" * 64)
        with self.assertRaisesRegex(updater.UpdateError, "checksum.*wasn't changed"):
            self.download(url, info)
        self.assertEqual(len(self.server.requests), 1)
        self.assert_nothing_kept()


class ReleaseChoiceTests(unittest.TestCase):
    def details(self, **changes):
        asset = {"browser_download_url": updater.DOWNLOAD_PREFIX + "v1/x", "size": 5,
                 "digest": "sha256:" + "a" * 64, **changes}
        return updater._asset_details(asset)

    def test_a_verifiable_official_asset_is_used(self):
        api = updater.API_ASSET_PREFIX + "1"
        self.assertEqual(self.details(url=api), {"url": updater.DOWNLOAD_PREFIX + "v1/x", "api_url": api,
                                                 "size": 5, "sha256": "a" * 64})

    def test_an_api_link_for_anything_else_is_ignored(self):
        details = self.details(url="https://api.github.com/repos/someone/else/releases/assets/1")
        self.assertIsNone(details["api_url"])
        self.assertIsNotNone(details["url"])

    def test_anything_unverifiable_falls_back_to_the_page(self):
        for changes in ({"digest": None}, {"digest": "md5:abc"}, {"size": 0},
                        {"browser_download_url": "https://github.com/someone/else/releases/download/v1/x"}):
            with self.subTest(changes):
                self.assertIsNone(self.details(**changes)["url"])
        self.assertIsNone(updater._asset_details(None)["url"])


class _Response(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}


class HandoffTests(unittest.TestCase):
    NEW = b"new version"

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rbo-update-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.exe = os.path.join(self.dir, "Rekordbox Organizer.exe")
        with open(self.exe, "wb") as f:
            f.write(b"old version")
        self.info = info_for(updater.DOWNLOAD_PREFIX + "v9/x.exe", self.NEW)
        self.started = mock.Mock()
        for p in (mock.patch.object(sys, "executable", self.exe),
                  mock.patch.object(sys, "platform", "win32"),
                  mock.patch.object(updater, "_open", lambda url, timeout: _Response(self.NEW)),
                  mock.patch.object(updater.subprocess, "Popen", self.started)):
            p.start()
            self.addCleanup(p.stop)

    def read_exe(self):
        with open(self.exe, "rb") as f:
            return f.read()

    def assert_only_the_old_app(self):
        self.assertEqual(self.read_exe(), b"old version")
        self.assertEqual(os.listdir(self.dir), [os.path.basename(self.exe)])

    def test_a_verified_download_is_swapped_in_and_started(self):
        updater.install(self.info, lambda d, t: None)
        self.assertEqual(self.read_exe(), self.NEW)
        self.assertEqual(self.started.call_args.args, ([self.exe],))
        self.assertEqual(self.started.call_args.kwargs["env"]["PYINSTALLER_RESET_ENVIRONMENT"], "1")
        self.assertEqual(sorted(os.listdir(self.dir)),
                         sorted(map(os.path.basename, (self.exe, self.exe + ".old"))))

    def test_a_bad_download_never_touches_the_app(self):
        with self.assertRaises(updater.UpdateError):
            updater.install(dict(self.info, sha256="0" * 64), lambda d, t: None)
        self.assert_only_the_old_app()
        self.started.assert_not_called()

    def test_a_failed_swap_puts_the_current_version_back(self):
        real = os.replace

        def new_file_stuck(a, b):
            if a == self.exe + ".new":
                raise OSError("in use")
            return real(a, b)
        with mock.patch.object(updater.os, "replace", new_file_stuck), \
             self.assertRaisesRegex(updater.UpdateError, "in use.*wasn't changed"):
            updater.install(self.info, lambda d, t: None)
        self.assert_only_the_old_app()
        self.started.assert_not_called()

    def test_a_new_version_that_cant_start_is_rolled_back(self):
        self.started.side_effect = OSError("blocked by policy")
        with self.assertRaisesRegex(updater.UpdateError, "wouldn't start.*blocked by policy.*wasn't changed"):
            updater.install(self.info, lambda d, t: None)
        self.assert_only_the_old_app()

    def test_leftovers_are_cleared_at_next_start(self):
        for suffix in (".old", ".new", ".new.part"):
            with open(self.exe + suffix, "wb"):
                pass
        with mock.patch.object(sys, "frozen", True, create=True):
            updater.cleanup_old()
        self.assert_only_the_old_app()


class WindowsEdgeCaseTests(HandoffTests):
    def setUp(self):
        super().setUp()
        sleeping = mock.patch.object(updater.time, "sleep")
        self.sleep = sleeping.start()
        self.addCleanup(sleeping.stop)

    def replacing(self, fail):
        real = os.replace

        def replace(a, b):
            problem = fail(a, b)
            if problem:
                raise problem
            return real(a, b)
        return mock.patch.object(updater.os, "replace", replace)

    def test_a_leftover_old_copy_doesnt_stop_the_update(self):
        with open(self.exe + ".old", "wb") as f:
            f.write(b"older still")
        updater.install(self.info, lambda d, t: None)
        with open(self.exe + ".old", "rb") as f:
            self.assertEqual(f.read(), b"old version")

    def test_a_virus_scanner_briefly_holding_the_new_file_is_waited_out(self):
        attempts = []

        def scanned_twice(a, b):
            if a == self.exe + ".new":
                attempts.append(a)
                if len(attempts) < 3:
                    return PermissionError("scanning")
        with self.replacing(scanned_twice):
            updater.install(self.info, lambda d, t: None)
        self.assertEqual(self.read_exe(), self.NEW)
        self.assertEqual(len(attempts), 3)
        self.started.assert_called_once()

    def test_a_file_that_stays_locked_gives_up_and_restores(self):
        with self.replacing(lambda a, b: PermissionError("locked") if a == self.exe + ".new" else None), \
             self.assertRaisesRegex(updater.UpdateError, "locked.*wasn't changed"):
            updater.install(self.info, lambda d, t: None)
        self.assert_only_the_old_app()
        self.assertEqual(self.sleep.call_count, len(updater.REPLACE_WAITS))
        self.started.assert_not_called()

    def test_the_running_app_being_locked_changes_nothing(self):
        with self.replacing(lambda a, b: OSError("no rename") if a == self.exe else None), \
             self.assertRaisesRegex(updater.UpdateError, "wasn't changed"):
            updater.install(self.info, lambda d, t: None)
        self.assert_only_the_old_app()

    def test_a_failed_rollback_says_where_the_old_version_is(self):
        def stuck(a, b):
            if a == self.exe + ".old":
                return OSError("stuck")
        self.started.side_effect = OSError("blocked")
        with self.replacing(stuck), self.assertRaises(updater.UpdateError) as caught:
            updater.install(self.info, lambda d, t: None)
        self.assertIn(self.exe + ".old", str(caught.exception))
        self.assertNotIn("wasn't changed", str(caught.exception))
        with open(self.exe + ".old", "rb") as f:
            self.assertEqual(f.read(), b"old version")

    def test_a_full_disk_leaves_nothing_behind(self):
        with mock.patch.object(updater, "_download", side_effect=OSError("disk full")), \
             self.assertRaisesRegex(updater.UpdateError, "disk full.*wasn't changed"):
            updater.install(self.info, lambda d, t: None)
        self.assert_only_the_old_app()
        self.started.assert_not_called()


class MacHandoffTests(unittest.TestCase):
    """The real swap on real folders; only ditto and open are stood in for, unless on a Mac."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rbo-update-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.app = os.path.join(self.dir, "Rekordbox Organizer.app")
        self.program = os.path.join(self.app, "Contents", "MacOS", "Rekordbox Organizer")
        os.makedirs(os.path.dirname(self.program))
        with open(self.program, "w") as f:
            f.write("old")
        self.bundle = self.zipped({"Rekordbox Organizer.app/Contents/MacOS/Rekordbox Organizer": "new"})
        self.launches = []
        self.launch_problem = None
        self.unpack_problem = None
        self.run_commands = mock.Mock(side_effect=self.fake_run)
        for p in (mock.patch.object(sys, "executable", self.program),
                  mock.patch.object(sys, "platform", "darwin"),
                  mock.patch.object(updater, "_open", lambda url, timeout: _Response(self.bundle)),
                  mock.patch.object(updater.subprocess, "run", self.run_commands),
                  mock.patch.object(updater.subprocess, "Popen", side_effect=AssertionError("no Popen"))):
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def zipped(files):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z:
            for name, text in files.items():
                z.writestr(name, text)
        return out.getvalue()

    def fake_run(self, command, **kwargs):
        if command[0] == "ditto":
            if self.unpack_problem:
                raise self.unpack_problem
            try:
                with zipfile.ZipFile(command[3]) as z:
                    z.extractall(command[4])
            except zipfile.BadZipFile as exc:
                raise subprocess.CalledProcessError(1, command) from exc
            return
        self.assertEqual(command[:2], ["open", "-n"])
        self.assertEqual(kwargs["env"]["PYINSTALLER_RESET_ENVIRONMENT"], "1")
        self.assertTrue(kwargs["check"])
        work = next(n for n in os.listdir(self.dir) if not n.endswith(".app"))
        self.launches.append(os.listdir(os.path.join(self.dir, work)))
        if self.launch_problem:
            raise self.launch_problem

    def info(self, payload=None):
        return info_for(updater.DOWNLOAD_PREFIX + "v9/x.zip", payload or self.bundle)

    def install(self):
        updater.install(self.info(), lambda d, t: None)

    def running_program(self):
        with open(self.program) as f:
            return f.read()

    def assert_only_the_old_app(self):
        self.assertEqual(self.running_program(), "old")
        self.assertEqual(os.listdir(self.dir), [os.path.basename(self.app)])

    def commands(self):
        return [c.args[0][0] for c in self.run_commands.call_args_list]

    def test_success_swaps_then_launches_and_only_then_removes_the_old_app(self):
        self.install()
        self.assertEqual(self.running_program(), "new")
        self.assertEqual(os.listdir(self.dir), [os.path.basename(self.app)])
        self.assertEqual(self.commands(), ["ditto", "open"])
        self.assertIn("previous.app", self.launches[0])

    def test_a_bad_checksum_is_never_unpacked(self):
        with self.assertRaisesRegex(updater.UpdateError, "checksum"):
            updater.install(dict(self.info(), sha256="0" * 64), lambda d, t: None)
        self.assert_only_the_old_app()
        self.assertEqual(self.commands(), [])

    def test_extraction_failing_changes_nothing(self):
        for problem in (subprocess.CalledProcessError(1, "ditto"), FileNotFoundError("ditto")):
            with self.subTest(problem):
                self.unpack_problem = problem
                with self.assertRaisesRegex(updater.UpdateError, "unpacked.*wasn't changed"):
                    self.install()
                self.assert_only_the_old_app()

    def test_an_archive_that_wont_open_changes_nothing(self):
        self.bundle = b"this is not a zip"
        with self.assertRaisesRegex(updater.UpdateError, "unpacked"):
            self.install()
        self.assert_only_the_old_app()

    def test_an_archive_without_the_program_changes_nothing(self):
        for files in ({"readme.txt": "x"}, {"Rekordbox Organizer.app/Contents/Info.plist": "x"}):
            with self.subTest(files):
                self.bundle = self.zipped(files)
                with self.assertRaisesRegex(updater.UpdateError, "didn't contain"):
                    self.install()
                self.assert_only_the_old_app()

    def test_moving_the_old_app_failing_changes_nothing(self):
        real = os.rename

        def refuse(a, b):
            if a == self.app:
                raise PermissionError("not allowed")
            return real(a, b)
        with mock.patch.object(updater.os, "rename", refuse), \
             self.assertRaisesRegex(updater.UpdateError, "not allowed.*wasn't changed"):
            self.install()
        self.assert_only_the_old_app()

    def test_a_failed_swap_rolls_back(self):
        real = os.rename

        def refuse(a, b):
            if b == self.app and not a.endswith("previous.app"):
                raise OSError("no room")
            return real(a, b)
        with mock.patch.object(updater.os, "rename", refuse), \
             self.assertRaisesRegex(updater.UpdateError, "no room.*wasn't changed"):
            self.install()
        self.assert_only_the_old_app()
        self.assertEqual(self.commands(), ["ditto"])

    def test_a_launch_failure_rolls_back(self):
        for problem in (OSError("no open"), subprocess.CalledProcessError(1, "open"),
                        subprocess.TimeoutExpired("open", 60)):
            with self.subTest(problem):
                self.launch_problem = problem
                with self.assertRaisesRegex(updater.UpdateError, "wouldn't start.*wasn't changed"):
                    self.install()
                self.assert_only_the_old_app()

    def test_a_failed_rollback_keeps_the_only_copy_and_says_where(self):
        self.launch_problem = OSError("no open")
        real = os.rename

        def stuck(a, b):
            if a.endswith("previous.app"):
                raise OSError("stuck")
            return real(a, b)
        with mock.patch.object(updater.os, "rename", stuck), self.assertRaises(updater.UpdateError) as caught:
            self.install()
        message = str(caught.exception)
        self.assertNotIn("wasn't changed", message)
        saved = message.split("saved at: ")[1]
        self.assertTrue(saved.endswith("previous.app"))
        with open(os.path.join(saved, "Contents", "MacOS", "Rekordbox Organizer")) as f:
            self.assertEqual(f.read(), "old")

    def test_a_translocated_app_is_refused_before_downloading(self):
        with mock.patch.object(updater.os.path, "abspath", lambda p: "/private/AppTranslocation/x/App.app"), \
             mock.patch.object(updater, "_download") as download, \
             self.assertRaisesRegex(updater.UpdateError, "Applications"):
            self.install()
        download.assert_not_called()

    @unittest.skipUnless(os.path.exists("/usr/bin/ditto"), "needs macOS's own ditto")
    def test_the_real_ditto_unpacks_the_archive(self):
        def real_ditto(command, **kwargs):
            return REAL_RUN(command, **kwargs) if command[0] == "ditto" else self.fake_run(command, **kwargs)
        self.run_commands.side_effect = real_ditto
        self.install()
        self.assertEqual(self.running_program(), "new")


class MoreRetryTests(unittest.TestCase):
    INFO = info_for(updater.DOWNLOAD_PREFIX + "v9/x")

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="rbo-update-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        sleeping = mock.patch.object(updater.time, "sleep")
        self.sleep = sleeping.start()
        self.addCleanup(sleeping.stop)
        self.messages = []

    def download(self):
        updater._download(self.INFO, os.path.join(self.dir, "a"), lambda d, t: None, self.messages.append)

    def test_a_bad_certificate_is_not_retried(self):
        opened = mock.Mock(side_effect=urllib.error.URLError(ssl.SSLCertVerificationError("bad cert")))
        with mock.patch.object(updater, "_open", opened), self.assertRaises(updater.UpdateError):
            self.download()
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(os.listdir(self.dir), [])

    def test_timeouts_are_retried_and_described_plainly(self):
        opened = mock.Mock(side_effect=urllib.error.URLError(TimeoutError("timed out")))
        with mock.patch.object(updater, "_open", opened), \
             self.assertRaisesRegex(updater.UpdateError, "after 4 tries.*connection timed out"):
            self.download()
        self.assertEqual(opened.call_count, 4)

    def test_the_label_moves_on_once_the_retry_starts(self):
        answers = [urllib.error.URLError(TimeoutError("timed out")), _Response(PAYLOAD)]

        def opened(url, timeout):
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        with mock.patch.object(updater, "_open", opened):
            self.download()
        self.assertRegex(self.messages[0], "Trying again in 2 seconds")
        self.assertEqual(self.messages[1], "The download: try 2 of 4 ...")

    def test_a_malformed_length_header_falls_back_to_the_published_size(self):
        response = _Response(PAYLOAD)
        response.headers = {"Content-Length": "lots"}
        with mock.patch.object(updater, "_open", lambda url, timeout: response):
            self.download()
        self.assertEqual(os.path.getsize(os.path.join(self.dir, "a")), len(PAYLOAD))


class RedirectEdgeTests(DownloadCase):
    def test_a_redirect_loop_stops(self):
        url = self.server.base + "/file"
        self.serve((302, {"Location": url}, b""))
        with self.assertRaises(updater.UpdateError):
            self.download(url)
        self.assertLess(len(self.server.requests), 12)
        self.assertEqual(os.listdir(self.dir), [])
        self.sleep.assert_not_called()


class RedirectRuleTests(unittest.TestCase):
    def follow(self, target):
        request = urllib.request.Request("https://github.com/a")
        return updater._TrustedRedirects().redirect_request(request, None, 302, "Found", {}, target)

    def test_real_handler_accepts_github_storage(self):
        self.assertIsNotNone(self.follow("https://release-assets.githubusercontent.com/x?sig=1"))

    def test_real_handler_refuses_everything_else(self):
        for target in ("http://release-assets.githubusercontent.com/x", "https://example.com/x",
                       "https://github.com:abc/x"):
            with self.subTest(target), self.assertRaises(updater.UpdateError):
                self.follow(target)


class ReleaseMetadataTests(unittest.TestCase):
    """check() on a response shaped like GitHub's, for each platform's asset."""

    def asset(self, name, **changes):
        return {"url": updater.API_ASSET_PREFIX + "7", "name": name, "state": "uploaded", "size": 10,
                "digest": "sha256:" + "b" * 64, "content_type": "application/octet-stream",
                "browser_download_url": updater.DOWNLOAD_PREFIX + "v9.0.0/" + name, **changes}

    def check(self, platform="win32", **changes):
        release = {"tag_name": "v9.0.0", "body": "Notes", "draft": False, "prerelease": False,
                   "html_url": "https://github.com/x/releases/tag/v9.0.0",
                   "assets": [self.asset(n, **changes) for n in updater.ASSETS.values()]}
        with mock.patch.object(sys, "platform", platform), mock.patch.object(updater, "VERSION", "1.0.0"), \
             mock.patch.object(updater, "_open", lambda url, timeout: _Response(json.dumps(release).encode())):
            return updater.check()

    def test_each_platform_gets_its_own_verified_asset(self):
        for platform, name in updater.ASSETS.items():
            with self.subTest(platform):
                found = self.check(platform)
                self.assertTrue(found["url"].endswith("/" + name))
                self.assertEqual((found["api_url"], found["sha256"], found["size"]),
                                 (updater.API_ASSET_PREFIX + "7", "b" * 64, 10))

    def test_no_published_digest_goes_to_the_page_and_keeps_it(self):
        for changes in ({"digest": None}, {"digest": ""}, {"state": "starter"}, {"digest": "sha256:short"}):
            with self.subTest(changes):
                found = self.check(**changes)
                self.assertEqual(found["version"], "9.0.0")
                self.assertIsNone(found["url"])
                self.assertIsNone(found["api_url"])
                self.assertEqual(found["page"], "https://github.com/x/releases/tag/v9.0.0")

    def test_no_digest_never_reaches_install_in_the_app(self):
        found = self.check(digest=None)
        window = mock.MagicMock(new_release=found)
        with mock.patch.object(app.updater, "install") as install, mock.patch("webbrowser.open") as opened:
            app.App.update_app(window)
        install.assert_not_called()
        opened.assert_called_once_with(found["page"])


class AppHandlingTests(unittest.TestCase):
    """App.update_app on a stand-in window, so no real app or library is opened."""

    INFO = {"version": "9.0.0", "notes": "", "url": "https://x", "sha256": "a", "size": 1, "page": "https://page"}

    def run_update(self, install, info=None, answer=True):
        window = mock.MagicMock(new_release=info or self.INFO)
        pending = []
        window.after = lambda ms, function: pending.append(function)
        with mock.patch.object(app.updater, "install", install), \
             mock.patch.object(app.messagebox, "askokcancel", return_value=True), \
             mock.patch.object(app.messagebox, "askyesno", return_value=answer) as self.asked, \
             mock.patch("webbrowser.open") as self.opened:
            app.App.update_app(window)
            deadline = time.time() + 10
            while pending and time.time() < deadline:
                pending.pop(0)()
                time.sleep(0.005)
        self.assertEqual(pending, [])
        return window

    def test_failure_explains_itself_and_offers_the_download_page(self):
        def fail(info, progress, notify):
            notify("Trying again")
            raise updater.UpdateError("still failed after 4 tries (HTTP 504)")
        window = self.run_update(fail)
        self.assertIn("HTTP 504", self.asked.call_args.args[1])
        self.opened.assert_called_once_with("https://page")
        window.update_btn.configure.assert_called_with(state="normal")
        window.update_label.configure.assert_any_call(text="Trying again")
        window.master.destroy.assert_not_called()

    def test_declining_the_download_page_opens_nothing(self):
        def fail(info, progress, notify):
            raise updater.UpdateError("no")
        self.run_update(fail, answer=False)
        self.opened.assert_not_called()

    def test_success_hands_over_to_the_new_version(self):
        received = []
        window = self.run_update(lambda info, progress, notify: received.append(info))
        self.assertEqual(received, [self.INFO])
        window.master.destroy.assert_called_once()

    def test_an_unverifiable_release_goes_straight_to_the_page(self):
        install = mock.Mock()
        self.run_update(install, dict(self.INFO, url=None))
        install.assert_not_called()
        self.opened.assert_called_once_with("https://page")


if __name__ == "__main__":
    unittest.main()
