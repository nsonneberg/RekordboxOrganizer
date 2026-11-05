"""Optional release metadata (genres / styles) from online databases.

Providers:

* MusicBrainz (musicbrainz.org/ws/2, no account needed). Rules: at most about
  one request per second per IP, and a User-Agent naming the app and a contact
  (https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting). Its genre and
  tag associations are "supplementary data" under CC BY-NC-SA 3.0
  (https://musicbrainz.org/doc/MusicBrainz_Database).
* Discogs (api.discogs.com), off unless switched on separately. Search works
  without a token at 25 requests a minute (60 with a personal access token
  from discogs.com/settings/developers); the app paces itself well under that.
  Styles are release-level. Discogs' API Terms of Use (as reported, updated
  2025-05-27; the official page couldn't be read from here) don't allow
  showing content more than 6 hours behind discogs.com or keeping it longer
  than necessary, and ask for "Data provided by Discogs" attribution; its
  Terms of Service restrict using its content to develop machine-learning
  software. So Discogs answers are cached for at most 6 hours (Discogs.cache
  _hours), shown with that attribution, and never used to fit the app's
  weights (genres._trainable_metadata).

Matching is deliberately conservative (match()):
- the title must match closely once version text and junk are removed;
- the artist must overlap (if the track has one);
- the version must agree: "(Tomas Reed Remix)" only matches a release that names
  Tomas Reed; a plain title doesn't match a remix and vice versa ("Original /
  Extended Mix" and "Radio Edit" count as the original);
- a length more than LENGTH_REJECT seconds off rejects the candidate;
- if the two best candidates are about as good but disagree on genre, nothing
  is used (ambiguous).
Genres come from the matched recording, else its release group / release,
at a lower quality. Artist-level genres are never used. Compilations by
"Various" are skipped.

Results (and "no match") are cached in metadata_cache.json with an expiry
per provider; network problems aren't cached. offline=True uses the cache
only and sends nothing. After MAX_FAILURES failures in a row a
provider is skipped for the rest of the run, so working offline costs a few
seconds at most.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

CACHE_VERSION = 1
MATCH_DAYS, MISS_DAYS = 180, 30       # MusicBrainz cache expiry for found / not found
MAX_FAILURES = 3
TITLE_MIN = 0.88                      # title similarity needed
ARTIST_MIN = 0.5                      # artist token overlap needed (when the track has an artist)
LENGTH_OK, LENGTH_REJECT = 4, 15      # seconds
AMBIGUOUS = 0.08                      # best two candidates closer than this are ambiguous
LEVEL_QUALITY = {"recording": 1.0, "release-group": 0.7, "release": 0.7}

USER_AGENT = "RekordboxOrganizer/{version} ( https://github.com/nsonneberg/RekordboxOrganizer )"


class ProviderError(Exception):
    """The service couldn't be reached or answered with an error (not "no match")."""


@dataclass
class Query:
    artist: str
    title: str          # without version / junk
    version: str        # "Tomas Reed Remix", "" for the original
    length: float       # seconds, 0 if unknown


@dataclass
class MetadataResult:
    source: str                      # provider name
    genres: list                     # [(genre or style, weight)]
    quality: float                   # 0-1: match quality x level
    level: str                       # recording / release-group / release
    describe: str                    # what matched, for the review window
    match_id: str = ""
    parts: list = field(default_factory=list)   # each provider's own result, when merged

    def to_json(self) -> dict:
        data = asdict(self)
        data.pop("parts", None)
        return data

    @classmethod
    def from_json(cls, data: dict) -> "MetadataResult":
        return cls(data["source"], [tuple(g) for g in data["genres"]], float(data["quality"]),
                   data["level"], data["describe"], data.get("match_id", ""))


# --------------------------------------------------------------------------- #
# Parsing and matching
# --------------------------------------------------------------------------- #

_BRACKET = re.compile(r"[\(\[]([^\)\]]*)[\)\]]")
VERSION_WORDS = ("remix", "edit", "bootleg", "flip", "rework", "vip", "mashup", "refix",
                 "remake", "cover", "dub", "mix", "version", "rmx")
ORIGINAL = ("original mix", "extended mix", "radio edit", "original", "extended", "club mix",
            "radio mix", "extended version", "original version")
JUNK = re.compile(r"\b(free (dl|download)|out now|premiere|clip|preview|hq|official)\b", re.I)


def _norm(text: str) -> str:
    text = text.casefold().replace("&", " and ")
    text = re.sub(r"\b(feat|ft|featuring)\b\.?.*", " ", text)
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def parse(name: str, artist: str = "", length: float = 0.0) -> Query:
    """A rekordbox title (and artist) -> Query. Handles "Artist - Title (X Remix)"."""
    version = ""
    for part in _BRACKET.findall(name):
        low = part.casefold().strip()
        if JUNK.search(low) or not low:
            continue
        if low in ORIGINAL:
            continue
        if any(re.search(rf"\b{w}\b", low) for w in VERSION_WORDS):
            version = part.strip()
    title = _BRACKET.sub(" ", name)
    if not artist and " - " in title:
        artist, title = title.split(" - ", 1)
    elif " - " in title and _norm(title.split(" - ", 1)[0]) == _norm(artist):
        title = title.split(" - ", 1)[1]
    title = JUNK.sub(" ", title)
    return Query(" ".join(artist.split()), " ".join(title.split()), version, float(length or 0))


def _artist_overlap(a: str, b: str) -> float:
    ta, tb = set(_norm(a).split()) - {"and", "x", "vs"}, set(_norm(b).split()) - {"and", "x", "vs"}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _version_of(title: str) -> str:
    return parse(title).version


def match(query: Query, cand_artist: str, cand_title: str, cand_length: float = 0.0) -> float:
    """How well a candidate recording matches (0 = reject)."""
    cand = parse(cand_title, cand_artist)
    title_sim = difflib.SequenceMatcher(None, _norm(query.title), _norm(cand.title)).ratio()
    if title_sim < TITLE_MIN:
        return 0.0
    score = title_sim
    if query.artist:
        overlap = _artist_overlap(query.artist, cand_artist)
        if overlap < ARTIST_MIN:
            return 0.0
        score *= 0.7 + 0.3 * overlap
    else:
        score *= 0.6                 # no artist to check: never a top-quality match
    qv, cv = _norm(query.version), _norm(cand.version)
    if bool(qv) != bool(cv):
        return 0.0                   # remix vs original: a different recording
    if qv:
        q_words = set(qv.split()) - set(VERSION_WORDS)
        c_words = set(cv.split()) - set(VERSION_WORDS)
        if q_words and not (q_words & c_words):
            return 0.0               # someone else's remix
    if query.length and cand_length:
        off = abs(query.length - cand_length)
        if off > LENGTH_REJECT:
            return 0.0
        score *= 1.0 if off <= LENGTH_OK else 0.85
    else:
        score *= 0.9
    return round(score, 4)


def best_match(query: Query, candidates: list) -> Optional[tuple]:
    """candidates: [(score, genres, ...)] -> the best one, or None if none / ambiguous."""
    ranked = sorted((c for c in candidates if c[0] > 0), key=lambda c: -c[0])
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < AMBIGUOUS:
        a = {g.casefold() for g, _ in ranked[0][1]}
        b = {g.casefold() for g, _ in ranked[1][1]}
        if a and b and not (a & b):
            return None              # equally good matches that disagree: don't guess
    return ranked[0]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def _http_json(url: str, headers: dict, timeout: float = 15) -> dict:
    import ssl
    import urllib.error
    import urllib.request
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, context=context, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {}
        raise ProviderError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ProviderError(str(getattr(exc, "reason", exc))) from exc


class Provider:
    """Base class: pacing, retries and failure counting around fetch()."""
    name = "?"
    interval = 1.0                   # seconds between requests
    match_hours = MATCH_DAYS * 24    # how long an answer may be cached
    miss_hours = MISS_DAYS * 24
    attribution = ""                 # shown next to its data

    def __init__(self, fetch: Optional[Callable[[str, dict], dict]] = None,
                 sleep: Callable[[float], None] = time.sleep, version: str = "dev"):
        self.fetch = fetch or _http_json
        self.sleep = sleep
        self.headers = {"User-Agent": USER_AGENT.format(version=version),
                        "Accept": "application/json"}
        self._last = 0.0
        self._lock = threading.Lock()
        self.failures = 0

    @property
    def disabled(self) -> bool:
        return self.failures >= MAX_FAILURES

    def get(self, url: str) -> dict:
        if self.disabled:
            raise ProviderError("skipped after repeated failures")
        for attempt in range(3):
            with self._lock:
                wait = self._last + self.interval - time.monotonic()
                if wait > 0:
                    self.sleep(wait)
                self._last = time.monotonic()
            try:
                data = self.fetch(url, self.headers)
                self.failures = 0
                return data
            except ProviderError as exc:
                if "503" in str(exc) or "429" in str(exc):
                    self.sleep(self.interval * (attempt + 2))      # asked to slow down
                    continue
                self.failures += 1
                raise
        self.failures += 1
        raise ProviderError("the service kept asking to slow down")

    def lookup(self, query: Query) -> Optional[MetadataResult]:
        raise NotImplementedError


class MusicBrainz(Provider):
    name = "MusicBrainz"
    interval = 1.1
    attribution = "Genres from MusicBrainz (CC BY-NC-SA 3.0)"
    base = "https://musicbrainz.org/ws/2"

    def lookup(self, query: Query) -> Optional[MetadataResult]:
        if not query.title:
            return None
        terms = [f'recording:"{_lucene(query.title)}"']
        if query.artist:
            terms.append(f'artist:"{_lucene(query.artist)}"')
        url = (f"{self.base}/recording?fmt=json&limit=10&query="
               + urllib.parse.quote(" AND ".join(terms)))
        data = self.get(url)
        candidates = []
        for rec in data.get("recordings", []):
            artist = "".join(c.get("name", "") + c.get("joinphrase", "")
                             for c in rec.get("artist-credit", []))
            title = rec.get("title", "")
            if rec.get("disambiguation"):
                title = f"{title} ({rec['disambiguation']})"
            score = match(query, artist, title, (rec.get("length") or 0) / 1000)
            if score:
                tags = [(t["name"], t.get("count", 1)) for t in rec.get("tags", [])]
                candidates.append((score, tags, rec, artist))
        best = best_match(query, candidates)
        if best is None:
            return None
        score, _, rec, artist = best
        detail = self.get(f"{self.base}/recording/{rec['id']}?fmt=json&inc=genres+tags+releases"
                          "+release-groups")
        genres = _counted(detail.get("genres")) or _counted(detail.get("tags"))
        level = "recording"
        describe = f"{artist} - {rec.get('title', '')}"
        if not genres:
            groups = sorted({r.get("release-group", {}).get("id") for r in detail.get("releases", [])
                             if r.get("release-group", {}).get("id")
                             and r.get("artist-credit-phrase", "").casefold() != "various artists"})
            for gid in groups[:2]:
                group = self.get(f"{self.base}/release-group/{gid}?fmt=json&inc=genres")
                if group.get("primary-type") == "Compilation":
                    continue
                genres = _counted(group.get("genres"))
                if genres:
                    level = "release-group"
                    describe += f" (release: {group.get('title', '')})"
                    break
        if not genres:
            return None
        return MetadataResult(self.name, genres, round(score * LEVEL_QUALITY[level], 4), level,
                              describe, rec["id"])


class Discogs(Provider):
    name = "Discogs"
    base = "https://api.discogs.com"
    match_hours = miss_hours = 6     # its terms: nothing more than 6 hours old
    attribution = "Data provided by Discogs"

    def __init__(self, token: str = "", **kw):
        super().__init__(**kw)
        self.interval = 1.2 if token else 2.6      # under 60 / 25 requests a minute
        if token:
            self.headers["Authorization"] = f"Discogs token={token}"

    def lookup(self, query: Query) -> Optional[MetadataResult]:
        if not query.title:
            return None
        params = {"type": "release", "track": query.title, "per_page": 10}
        if query.artist:
            params["artist"] = query.artist
        data = self.get(f"{self.base}/database/search?" + urllib.parse.urlencode(params))
        releases = [r for r in data.get("results", []) if "Various" not in r.get("title", "")]
        candidates = []
        for rel in releases[:3]:            # check the tracklists of the top releases
            detail = self.get(f"{self.base}/releases/{rel['id']}")
            if not detail:
                continue
            artists = " ".join(a.get("name", "") for a in detail.get("artists", []))
            if _norm(artists) in ("various",):
                continue
            styles = [(s, 1.0) for s in detail.get("styles", [])]
            styles += [(g, 0.3) for g in detail.get("genres", [])]
            for track in detail.get("tracklist", []):
                credit = " ".join(a.get("name", "") for a in track.get("artists", [])) or artists
                score = match(query, credit, track.get("title", ""),
                              _seconds(track.get("duration", "")))
                if score:
                    candidates.append((score, styles, detail, track, credit))
        best = best_match(query, candidates)
        if best is None or not best[1]:
            return None
        score, styles, detail, track, credit = best
        year = detail.get("year") or ""
        return MetadataResult(self.name, styles, round(score * LEVEL_QUALITY["release"], 4),
                              "release",
                              f"{credit} - {track.get('title', '')} (release: "
                              f"{detail.get('title', '')}{', ' + str(year) if year else ''})",
                              str(detail.get("id", "")))


def _lucene(text: str) -> str:
    return re.sub(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)', r"\\\1", text)


def _counted(items) -> list:
    return [(i["name"], max(1, int(i.get("count", 1)))) for i in (items or []) if i.get("name")]


def _seconds(duration: str) -> float:
    try:
        parts = [int(p) for p in duration.split(":")]
    except ValueError:
        return 0.0
    total = 0
    for p in parts:
        total = total * 60 + p
    return float(total)


# --------------------------------------------------------------------------- #
# Cache and batch lookup
# --------------------------------------------------------------------------- #

def cache_path() -> str:
    from organizer import app_data_dir
    return os.path.join(app_data_dir(), "metadata_cache.json")


def _cache_key(provider: str, q: Query) -> str:
    return "|".join([provider, _norm(q.artist), _norm(q.title), _norm(q.version),
                     str(int(round(q.length / 5) * 5))])


def load_cache(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["entries"] if data.get("version") == CACHE_VERSION else {}
    except (OSError, ValueError, KeyError):
        return {}


def save_cache(path: str, entries: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": CACHE_VERSION, "entries": entries}, f)
    os.replace(tmp, path)


@dataclass
class LookupStats:
    asked: int = 0
    matched: int = 0
    not_found: int = 0
    failed: int = 0
    from_cache: int = 0
    not_looked_up: int = 0      # offline and not in the cache
    by_provider: dict = field(default_factory=dict)   # provider -> matched count


def merge(results: list[MetadataResult]) -> Optional[MetadataResult]:
    """Several providers' matches for one track -> one result (genres weighted by quality)."""
    results = [r for r in results if r and r.genres]
    if not results:
        return None
    if len(results) == 1:
        return results[0]
    genres: dict[str, float] = {}
    for r in results:
        total = sum(w for _, w in r.genres) or 1
        for name, w in r.genres:
            genres[name] = genres.get(name, 0.0) + r.quality * w / total
    best = max(results, key=lambda r: r.quality)
    return MetadataResult(" + ".join(r.source for r in results), sorted(genres.items()),
                          best.quality, best.level, "; ".join(r.describe for r in results),
                          best.match_id, list(results))


def lookup_tracks(tracks, providers: list[Provider], log: Callable[[str], None] = print,
                  progress: Optional[Callable[[int, int], None]] = None,
                  cache_file: Optional[str] = None, now: Optional[float] = None,
                  offline: bool = False) -> tuple[dict[str, MetadataResult], LookupStats]:
    """{track id: MetadataResult} for the tracks that matched, plus counts.

    offline: answer from the cache only; nothing is sent anywhere.
    """
    cache_file = cache_file or cache_path()
    cache = load_cache(cache_file)
    now = time.time() if now is None else now
    out: dict[str, MetadataResult] = {}
    stats = LookupStats()
    changed = False
    for n, t in enumerate(tracks, 1):
        q = parse(t.name, t.artist, t.length)
        if not q.title:
            continue
        stats.asked += 1
        found, failed, unasked = [], False, 0
        for p in providers:
            key = _cache_key(p.name, q)
            hit = cache.get(key)
            hours = p.match_hours if hit and hit["result"] else p.miss_hours
            if hit and now - hit["time"] < hours * 3600:
                stats.from_cache += 1
                if hit["result"]:
                    found.append(MetadataResult.from_json(hit["result"]))
                continue
            if hit:                       # expired: don't keep it
                del cache[key]
                changed = True
            if offline:
                unasked += 1
                continue
            if p.disabled:
                failed = True
                continue
            try:
                result = p.lookup(q)
            except ProviderError:
                failed = True
                continue
            cache[key] = {"time": now, "result": result.to_json() if result else None}
            changed = True
            if result:
                found.append(result)
        merged = merge(found)
        if merged:
            out[t.id] = merged
            stats.matched += 1
            for r in found:
                stats.by_provider[r.source] = stats.by_provider.get(r.source, 0) + 1
        elif failed:
            stats.failed += 1
        elif unasked == len(providers):
            stats.not_looked_up += 1
        else:
            stats.not_found += 1
        if progress:
            progress(n, len(tracks))
    if changed:
        save_cache(cache_file, cache)
    log(f"Release info: {stats.matched} of {stats.asked} songs matched"
        + (f", {stats.not_found} not found" if stats.not_found else "")
        + (f", {stats.failed} couldn't be checked (offline?)" if stats.failed else "")
        + (f", {stats.not_looked_up} not looked up (offline mode)"
           if stats.not_looked_up else "") + ".")
    return out, stats
