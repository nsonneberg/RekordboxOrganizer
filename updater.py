"""Check GitHub for a newer release and install it in place.

Only the built apps update themselves; when running from source, use `git pull`.
  Windows: the running .exe can't be overwritten, but it can be renamed, so the
           new one is downloaded next to it and swapped in; the old one is
           deleted the next time the app starts.
  macOS:   the new .app is unzipped next to the old one and swapped in.
Either way the new version is then started and this one quits.

Network trouble is retried a few times. A download only counts once its size and
SHA-256 match what GitHub publishes for the release; until then nothing is replaced.
"""

from __future__ import annotations

import hashlib
import http.client
import itertools
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Optional

from version import VERSION

REPO = "nsonneberg/RekordboxOrganizer"
LATEST_API = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
DOWNLOAD_PREFIX = f"https://github.com/{REPO}/releases/download/"
API_ASSET_PREFIX = f"https://api.github.com/repos/{REPO}/releases/assets/"
ASSETS = {"win32": "Rekordbox-Organizer-Windows.exe", "darwin": "Rekordbox-Organizer-macOS.zip"}

RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}
RETRY_DELAYS = (2, 5, 15)           # seconds before the 2nd, 3rd and 4th try
CHECK_RETRY_DELAYS = (2, 5)
MAX_RETRY_AFTER = 30                # the most we'll wait when a server asks us to
REPLACE_WAITS = (0.2, 0.5, 1, 2)    # for a virus scanner holding a file that was just written
KEPT = " Your current version wasn't changed."


class UpdateError(RuntimeError):
    """The update failed; the message is written for the person using the app."""


class _IncompleteDownload(Exception):
    """The connection ended early, or gave a different amount than expected."""


_NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError,
                   http.client.HTTPException, ssl.SSLError, _IncompleteDownload)


def can_self_update() -> bool:
    return bool(getattr(sys, "frozen", False)) and sys.platform in ASSETS


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", text)[:3])


def _context() -> ssl.SSLContext:
    try:
        import certifi   # bundled apps may not see the system's certificates
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _is_trusted(url: str) -> bool:
    """GitHub over HTTPS only: release downloads redirect to its storage host."""
    try:
        parts = urllib.parse.urlsplit(url)
        host, port = (parts.hostname or "").casefold(), parts.port
    except ValueError:      # malformed, such as a bad port
        return False
    if parts.scheme != "https" or port not in (None, 443) or "@" in parts.netloc or "\\" in url:
        return False
    return host in ("github.com", "api.github.com") or host.endswith(".githubusercontent.com")


class _TrustedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _is_trusted(newurl):
            raise UpdateError("The download redirected somewhere unexpected, so it was stopped." + KEPT)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(url: str, timeout: float):
    if not _is_trusted(url):
        raise UpdateError("That download address isn't an official GitHub one." + KEPT)
    accept = "application/vnd.github+json" if url == LATEST_API else "application/octet-stream"
    request = urllib.request.Request(url, headers={"User-Agent": "RekordboxOrganizer/" + VERSION,
                                                   "Accept": accept})
    opener = urllib.request.build_opener(_TrustedRedirects,
                                         urllib.request.HTTPSHandler(context=_context()))
    return opener.open(request, timeout=timeout)


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRY_STATUSES
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    return not isinstance(reason, ssl.SSLCertVerificationError)


def _describe(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        host = urllib.parse.urlsplit(exc.url or "").hostname
        return f"HTTP {exc.code} {exc.reason}" + (f" from {host}" if host else "")
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(reason, TimeoutError):
        return "the connection timed out"
    return str(reason) or type(reason).__name__


def _wait_before_retry(exc: Exception, planned: int) -> int:
    """The planned delay, or longer (within reason) if the server said so."""
    if isinstance(exc, urllib.error.HTTPError):
        asked = exc.headers.get("Retry-After") if exc.headers else None
        if asked and asked.strip().isdigit():
            return max(planned, min(int(asked), MAX_RETRY_AFTER))
    return planned


def _with_retries(action, what: str, delays: tuple, notify: Callable[[str], None], after: str = ""):
    """Run action, retrying transient network errors. Raises UpdateError once it gives up."""
    tries = len(delays) + 1
    for attempt in range(1, tries + 1):
        try:
            return action()
        except _NETWORK_ERRORS as exc:
            problem = _describe(exc)
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            if not _is_transient(exc):
                raise UpdateError(f"{what} failed ({problem}).{after}") from exc
            if attempt == tries:
                raise UpdateError(f"{what} still failed after {tries} tries ({problem}).{after}") from exc
            wait = _wait_before_retry(exc, delays[attempt - 1])
            notify(f"{what} hit a problem ({problem}). Trying again in {wait} seconds "
                   f"(try {attempt + 1} of {tries}) ...")
            time.sleep(wait)
            notify(f"{what}: try {attempt + 1} of {tries} ...")


def _asset_details(asset: Optional[dict]) -> dict:
    """Where to get an asset and how to verify it; url is None if it can't be installed safely."""
    nothing = {"url": None, "api_url": None, "sha256": None, "size": None}
    if not asset:
        return nothing
    url, size = asset.get("browser_download_url") or "", asset.get("size")
    digest = re.fullmatch(r"sha256:([0-9a-f]{64})", asset.get("digest") or "")
    if not (url.startswith(DOWNLOAD_PREFIX) and digest and isinstance(size, int) and size > 0
            and asset.get("state", "uploaded") == "uploaded"):
        return nothing
    api_url = asset.get("url") or ""
    return {"url": url, "api_url": api_url if api_url.startswith(API_ASSET_PREFIX) else None,
            "sha256": digest.group(1), "size": size}


def check() -> Optional[dict]:
    """The newest release if it's newer than this app, else None. Raises UpdateError on network errors.

    "url" is None when there's no asset that can be verified: the app then offers the page.
    """
    def latest_release():
        with _open(LATEST_API, timeout=10) as response:
            return json.load(response)
    release = _with_retries(latest_release, "Checking for updates", CHECK_RETRY_DELAYS, lambda _: None)
    latest = release.get("tag_name", "")
    if not _version_tuple(latest) or _version_tuple(latest) <= _version_tuple(VERSION):
        return None
    asset = next((a for a in release.get("assets", [])
                  if a.get("name") == ASSETS.get(sys.platform)), None)
    return {"version": latest.lstrip("v"), "notes": release.get("body") or "",
            **_asset_details(asset), "page": release.get("html_url") or RELEASES_PAGE}


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _download(info: dict, dest: str, progress: Callable[[int, int], None],
              notify: Callable[[str], None]) -> None:
    """Fetch the asset to dest. dest only appears once it's complete and verified."""
    size, part = info["size"], dest + ".part"
    # GitHub's public download link can return 504s for a while as its API link keeps working.
    routes = itertools.cycle(filter(None, (info["url"], info.get("api_url"))))

    def fetch():
        digest = hashlib.sha256()
        done = 0
        progress(0, size)
        with _open(next(routes), timeout=30) as response, open(part, "wb") as out:
            length = str(response.headers.get("Content-Length") or "")
            announced = int(length) if length.isdigit() else 0
            while chunk := response.read(256 * 1024):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                progress(done, size)
        # http.client ends a cut-off body quietly here, so the count has to be checked.
        for expected in filter(None, (announced, size)):
            if done != expected:
                raise _IncompleteDownload(f"got {done:,} of {expected:,} bytes")
        return digest.hexdigest()

    try:
        actual = _with_retries(fetch, "The download", RETRY_DELAYS, notify, KEPT)
        if actual != info["sha256"]:
            raise UpdateError("The download didn't match the checksum GitHub published for "
                              "this release, so it was thrown away." + KEPT)
        os.replace(part, dest)
    finally:
        _remove(part)


def cleanup_old() -> None:
    """Delete what a Windows update left behind: the previous version and any unfinished download."""
    if can_self_update() and sys.platform == "win32":
        for suffix in (".old", ".new", ".new.part"):
            _remove(sys.executable + suffix)   # still closing? next time


def _fresh_environment() -> dict:
    """Our environment minus PyInstaller's own settings.

    A bundled app passes these on to programs it starts; the new version
    would then look for its files in this version's temporary folder, which
    is deleted when this one quits ("Failed to load Python DLL").
    """
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("_PYI_", "_MEIPASS")) and k != "TCL_LIBRARY" and k != "TK_LIBRARY"}
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return env


def _replace(source: str, target: str) -> None:
    """os.replace, waiting out a virus scanner that briefly holds a file open."""
    for wait in REPLACE_WAITS:
        try:
            return os.replace(source, target)
        except PermissionError:
            time.sleep(wait)
    os.replace(source, target)


def _couldnt_put_back(saved: str, exc: OSError) -> UpdateError:
    return UpdateError(f"The update failed and the previous version couldn't be put back ({exc}). "
                       f"It's saved at: {saved}")


def _install_windows(info: dict, progress, notify) -> None:
    exe = sys.executable
    new, old = exe + ".new", exe + ".old"

    def put_back():
        try:
            _replace(old, exe)
        except OSError as exc:
            raise _couldnt_put_back(old, exc) from exc

    try:
        _download(info, new, progress, notify)
        _replace(exe, old)              # a running .exe may be renamed, not overwritten
        try:
            _replace(new, exe)
        except OSError:
            put_back()
            raise
    except OSError as exc:
        raise UpdateError(f"The update couldn't be saved or put in place ({exc}).{KEPT}") from exc
    finally:
        _remove(new)
    try:
        subprocess.Popen([exe], close_fds=True, env=_fresh_environment())
    except OSError as exc:
        put_back()                      # it can't run here: keep the version that can
        raise UpdateError(f"The new version wouldn't start ({exc}).{KEPT}") from exc


def _install_mac(info: dict, progress, notify) -> None:
    # sys.executable is .../Name.app/Contents/MacOS/Name
    app = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..", ".."))
    if not app.endswith(".app") or "/AppTranslocation/" in app:
        raise UpdateError("Move Rekordbox Organizer into your Applications folder, open it "
                          "from there, and try again.")
    work, keep_work = None, False

    def put_back():
        nonlocal keep_work
        try:
            if os.path.lexists(app):
                os.rename(app, os.path.join(work, "failed.app"))
            os.rename(retired, app)
        except OSError as exc:
            keep_work = True            # it holds the only copy of the previous version
            raise _couldnt_put_back(retired, exc) from exc

    try:
        work = tempfile.mkdtemp(dir=os.path.dirname(app))   # same disk, so moves are instant
        archive = os.path.join(work, "update.zip")
        _download(info, archive, progress, notify)
        try:
            subprocess.run(["ditto", "-x", "-k", archive, work], check=True, capture_output=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            raise UpdateError("The download couldn't be unpacked." + KEPT) from exc
        fresh = next((os.path.join(work, n) for n in os.listdir(work) if n.endswith(".app")), None)
        program = os.path.basename(sys.executable)
        if not fresh or not os.path.isfile(os.path.join(fresh, "Contents", "MacOS", program)):
            raise UpdateError("The download didn't contain the app." + KEPT)
        retired = os.path.join(work, "previous.app")
        os.rename(app, retired)
        try:
            os.rename(fresh, app)
        except OSError:
            put_back()
            raise
        try:        # waits for open itself, so a refused launch can still be undone
            subprocess.run(["open", "-n", app], check=True, capture_output=True, timeout=60,
                           env=_fresh_environment())
        except (subprocess.SubprocessError, OSError) as exc:
            put_back()
            raise UpdateError("The new version wouldn't start." + KEPT) from exc
    except OSError as exc:
        raise UpdateError(f"The update couldn't be saved or put in place ({exc}).{KEPT}") from exc
    finally:
        # The old bundle can go once the new one is running; this process
        # keeps its open files until it quits.
        if work and not keep_work:
            shutil.rmtree(work, ignore_errors=True)


def install(info: dict, progress: Callable[[int, int], None],
            notify: Callable[[str], None] = lambda _: None) -> None:
    """Download and swap in the release from check(), then start it. The caller should quit.

    notify gets short messages to show while a download is being retried. Failures
    raise UpdateError, and leave the current version in place.
    """
    (_install_windows if sys.platform == "win32" else _install_mac)(info, progress, notify)
