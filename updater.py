"""Check GitHub for a newer release and install it in place.

Only the built apps update themselves; when running from source, use `git pull`.
  Windows: the running .exe can't be overwritten, but it can be renamed, so the
           new one is downloaded next to it and swapped in; the old one is
           deleted the next time the app starts.
  macOS:   the new .app is unzipped next to the old one and swapped in.
Either way the new version is then started and this one quits.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
from typing import Callable, Optional

from version import VERSION

REPO = "nsonneberg/RekordboxOrganizer"
LATEST_API = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
ASSETS = {"win32": "Rekordbox-Organizer-Windows.exe", "darwin": "Rekordbox-Organizer-macOS.zip"}


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


def _open(url: str, timeout: float):
    request = urllib.request.Request(url, headers={"User-Agent": "RekordboxOrganizer/" + VERSION,
                                                   "Accept": "application/vnd.github+json"})
    return urllib.request.urlopen(request, timeout=timeout, context=_context())


def check() -> Optional[dict]:
    """The newest release if it's newer than this app, else None. Raises on network errors."""
    with _open(LATEST_API, timeout=10) as response:
        release = json.load(response)
    latest = release.get("tag_name", "")
    if not _version_tuple(latest) or _version_tuple(latest) <= _version_tuple(VERSION):
        return None
    asset = next((a for a in release.get("assets", [])
                  if a.get("name") == ASSETS.get(sys.platform)), None)
    return {"version": latest.lstrip("v"), "notes": release.get("body") or "",
            "url": asset["browser_download_url"] if asset else None,
            "page": release.get("html_url") or RELEASES_PAGE}


def _download(url: str, dest: str, progress: Callable[[int, int], None]) -> None:
    with _open(url, timeout=30) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(256 * 1024):
            out.write(chunk)
            done += len(chunk)
            progress(done, total)


def cleanup_old() -> None:
    """Delete the previous version left behind by a Windows update."""
    if can_self_update() and sys.platform == "win32":
        old = sys.executable + ".old"
        if os.path.exists(old):
            try:
                os.remove(old)
            except OSError:
                pass   # still closing; next time


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


def install(url: str, progress: Callable[[int, int], None]) -> None:
    """Download and swap in the new version, then start it. The caller should quit."""
    if sys.platform == "win32":
        exe = sys.executable
        new, old = exe + ".new", exe + ".old"
        _download(url, new, progress)
        if os.path.exists(old):
            os.remove(old)
        os.replace(exe, old)            # a running .exe may be renamed, not overwritten
        try:
            os.replace(new, exe)
        except OSError:
            os.replace(old, exe)        # put the current version back
            raise
        subprocess.Popen([exe], close_fds=True, env=_fresh_environment())
        return

    # macOS: sys.executable is .../Name.app/Contents/MacOS/Name
    app = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..", ".."))
    if not app.endswith(".app") or "/AppTranslocation/" in app:
        raise RuntimeError("Move Rekordbox Organizer into your Applications folder, open it "
                           "from there, and try again.")
    work = tempfile.mkdtemp(dir=os.path.dirname(app))   # same disk, so moves are instant
    try:
        archive = os.path.join(work, "update.zip")
        _download(url, archive, progress)
        subprocess.run(["ditto", "-x", "-k", archive, work], check=True)
        fresh = next((os.path.join(work, n) for n in os.listdir(work) if n.endswith(".app")), None)
        if not fresh:
            raise RuntimeError("The download didn't contain the app.")
        retired = os.path.join(work, "previous.app")
        os.rename(app, retired)
        try:
            os.rename(fresh, app)
        except OSError:
            os.rename(retired, app)
            raise
    finally:
        # The old bundle can go once the new one is in place; the running
        # process keeps its open files until it quits.
        shutil.rmtree(work, ignore_errors=True)
    subprocess.Popen(["open", "-n", app], env=_fresh_environment())
