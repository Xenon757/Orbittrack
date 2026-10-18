"""Download, cache, and look up TLEs (two-line orbital elements) from CelesTrak."""

import time
import urllib.request
from pathlib import Path

from skyfield.api import EarthSatellite, load
from skyfield.iokit import parse_tle_file
from skyfield.timelib import Time

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"

# Cached files live next to the package, outside version control (see .gitignore).
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

# CelesTrak only updates most groups a few times a day and asks clients not to
# re-download more often than that, so a few hours of cache is both polite and fresh enough.
MAX_CACHE_AGE_HOURS = 6.0


def fetch_tle_file(
    group: str = "stations",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_age_hours: float = MAX_CACHE_AGE_HOURS,
    url_template: str = CELESTRAK_URL,
) -> Path:
    """Return a path to a local TLE file for `group`, downloading only if the cache is stale.

    `url_template` (with a {group} placeholder) lets a deployment read from a mirror of
    CelesTrak's files instead of CelesTrak itself.

    If the download fails but an old cached file exists, we fall back to it rather than
    crashing: slightly stale elements are still far better than no elements at all.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{group}.tle"

    if path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            return path

    url = url_template.format(group=group)
    request = urllib.request.Request(url, headers={"User-Agent": "OrbitTrack/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
    except OSError as error:
        if path.exists():
            print(f"warning: download failed ({error}); using cached {path.name}")
            return path
        raise

    # CelesTrak answers bad group names with a short plain-text error, not a TLE.
    # Every 3-line TLE record has a line starting with "1 " right after the name line.
    if b"\n1 " not in data:
        raise ValueError(f"{url} did not return TLE data for group {group!r}: {data[:100]!r}")

    path.write_bytes(data)
    return path


def load_satellites(path: Path) -> list[EarthSatellite]:
    """Parse a 3-line TLE file (name line + two element lines) into Skyfield satellites."""
    ts = load.timescale()
    with open(path, "rb") as f:
        return list(parse_tle_file(f, ts))


def find_satellite(satellites: list[EarthSatellite], query: str) -> EarthSatellite:
    """Find a satellite by NORAD catalog number (e.g. "25544") or exact name (case-insensitive).

    NORAD IDs are the reliable key; names are convenient but not guaranteed unique or stable.
    """
    query = query.strip()
    if query.isdigit():
        matches = [s for s in satellites if s.model.satnum == int(query)]
    else:
        matches = [s for s in satellites if s.name.strip().upper() == query.upper()]

    if not matches:
        raise LookupError(f"no satellite matching {query!r}")
    return matches[0]


def lookup_satellite(satellites: list[EarthSatellite], query: str) -> EarthSatellite:
    """find_satellite(), but on a miss suggest names that contain the query (e.g. "ISS")."""
    try:
        return find_satellite(satellites, query)
    except LookupError:
        close = [s.name.strip() for s in satellites if query.strip().upper() in s.name.upper()]
        hint = f" Did you mean: {', '.join(close[:5])}?" if close else ""
        raise LookupError(f"no satellite matching {query!r} in this TLE set.{hint}")


def tle_age_days(satellite: EarthSatellite, t: Time) -> float:
    """How many days `t` is from the TLE epoch. SGP4 error grows the further we get from it."""
    return float(t - satellite.epoch)
