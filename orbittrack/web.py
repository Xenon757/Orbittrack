"""Web interface: the same predictions as the CLI, served over HTTP.

Run locally with:
    uvicorn orbittrack.web:app --reload
"""

import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal, get_args
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi import Path as PathParam
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
import numpy as np
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.timelib import Time

from orbittrack.mapview import build_map
from orbittrack.observe import LookAngles, SubPoint, look_angles, sub_point
from orbittrack.passes import Pass, compass_direction, find_passes, sky_track
from orbittrack.pointing import ISS_DOWNLINK_HZ, PassProfile, pass_profile, pointing_csv
from orbittrack.station import WEST_LAFAYETTE, GroundStation
from orbittrack.accuracy import NASA_OEM_URL, Ephemeris, compare, fetch_oem_file, parse_oem, summary_rows
from orbittrack.tle import CELESTRAK_URL, fetch_tle_file, load_satellites, lookup_satellite, tle_age_days

# Only these CelesTrak groups can be requested. On a public server the group name ends up in
# a cache file path and in a request to CelesTrak, so accepting arbitrary text would allow
# path tricks ("../..") and let visitors make us hammer CelesTrak with junk requests.
Group = Literal["stations", "visual", "weather", "amateur", "science", "gps-ops"]
GROUP_LABELS: dict[str, str] = {
    "stations": "Space stations",
    "visual": "Brightest (visual)",
    "weather": "Weather",
    "amateur": "Amateur radio",
    "science": "Science",
    "gps-ops": "GPS (operational)",
}
assert set(GROUP_LABELS) == set(get_args(Group))

MAX_HOURS = 72  # each request does real orbit math; cap the work one visitor can ask for

app = FastAPI(title="OrbitTrack", description="Satellite pass predictions for a ground station.")
# Pages carry a day or more of position samples for the live display; numeric text like that
# compresses several-fold, so gzip anything over 1 KB for browsers that accept it.
app.add_middleware(GZipMiddleware, minimum_size=1000)
ts = load.timescale()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

DOWNLOAD_FAILED = "Couldn't download orbital data right now, and nothing is cached. Try again shortly."

# Where TLE files come from, as a URL with a {group} placeholder. Defaults to CelesTrak. Some
# hosting providers' networks can't reach CelesTrak, so a deployment can point this at a
# mirror (see .github/workflows/tle-mirror.yml), e.g. https://<user>.github.io/<repo>/{group}.tle
TLE_SOURCE = os.environ.get("ORBITTRACK_TLE_URL", CELESTRAK_URL)
# NASA's ISS ephemeris for the accuracy page; the deployment reads the same GitHub Pages mirror.
OEM_SOURCE = os.environ.get("ORBITTRACK_OEM_URL", NASA_OEM_URL)
log = logging.getLogger("orbittrack.web")


# --- TLE data shared by every visitor ---

_cache_lock = threading.Lock()
_parsed: dict[str, tuple[float, list[EarthSatellite]]] = {}  # group -> (file mtime, satellites)
_last_failure: dict[str, tuple[float, str]] = {}  # group -> (time.monotonic() of failure, reason)

# After a failed download, wait this long before trying CelesTrak again. Without it, every
# visitor would sit through the full network timeout and we'd keep retrying a failing server.
RETRY_AFTER_SECONDS = 300


def load_group(group: str) -> list[EarthSatellite]:
    """Satellites for a CelesTrak group, downloaded at most every few hours and parsed once.

    FastAPI runs requests on several threads. The lock makes sure that when the cache goes
    stale, only one request downloads a fresh copy while the others wait and reuse it.
    """
    with _cache_lock:
        failed = _last_failure.get(group)
        if failed and time.monotonic() - failed[0] < RETRY_AFTER_SECONDS:
            raise OSError(failed[1])
        try:
            path = fetch_tle_file(group, url_template=TLE_SOURCE)
        except (OSError, ValueError) as error:  # ValueError: the response wasn't TLE data
            reason = str(error) or type(error).__name__
            log.warning("TLE download for group %r from %s failed: %s", group, TLE_SOURCE, reason)
            _last_failure[group] = (time.monotonic(), reason)
            raise OSError(reason) from error
        _last_failure.pop(group, None)
        modified = path.stat().st_mtime
        if group not in _parsed or _parsed[group][0] != modified:
            _parsed[group] = (modified, load_satellites(path))
        return _parsed[group][1]


_ephemeris: tuple[float, Ephemeris] | None = None  # (file mtime, parsed ephemeris)


def load_iss_ephemeris() -> Ephemeris:
    """NASA's ISS ephemeris, downloaded at most every few hours and parsed once per download.

    Same lock and retry back-off as the TLE groups: a failed download is not retried for
    RETRY_AFTER_SECONDS, so visitors get an immediate 503 instead of waiting on the network.
    """
    global _ephemeris
    with _cache_lock:
        failed = _last_failure.get("nasa-oem")
        if failed and time.monotonic() - failed[0] < RETRY_AFTER_SECONDS:
            raise OSError(failed[1])
        try:
            path = fetch_oem_file(OEM_SOURCE)
            modified = path.stat().st_mtime
            if _ephemeris is None or _ephemeris[0] != modified:
                _ephemeris = (modified, parse_oem(path.read_text()))
        except (OSError, ValueError) as error:
            reason = str(error) or type(error).__name__
            log.warning("NASA ephemeris from %s unavailable: %s", OEM_SOURCE, reason)
            _last_failure["nasa-oem"] = (time.monotonic(), reason)
            raise OSError(reason) from error
        _last_failure.pop("nasa-oem", None)
        return _ephemeris[1]


def get_ephemeris_loader() -> Callable[[], Ephemeris]:
    """A FastAPI dependency, so tests can swap in the offline fixture ephemeris."""
    return load_iss_ephemeris


EphemerisLoader = Annotated[Callable[[], Ephemeris], Depends(get_ephemeris_loader)]


def get_group_loader() -> Callable[[str], list[EarthSatellite]]:
    """A FastAPI dependency, so tests can swap in the offline fixture file."""
    return load_group


GroupLoader = Annotated[Callable[[str], list[EarthSatellite]], Depends(get_group_loader)]


# --- request parameters ---

class PassQuery(BaseModel):
    """Query parameters shared by every page. FastAPI rejects out-of-range values with a 422."""

    sat: str = Field("25544", min_length=1, max_length=60, description="Satellite name or NORAD ID")
    group: Group = "stations"
    station: str = Field(WEST_LAFAYETTE.name, min_length=1, max_length=60)
    lat: float = Field(WEST_LAFAYETTE.latitude_deg, ge=-90, le=90)
    lon: float = Field(WEST_LAFAYETTE.longitude_deg, ge=-180, le=180)
    elev: float = Field(WEST_LAFAYETTE.elevation_m, ge=-500, le=9000, description="Metres above the ellipsoid")
    tz: str = Field(WEST_LAFAYETTE.timezone, description="IANA time zone for displayed times")
    start: datetime | None = Field(None, description="ISO 8601 start time; no offset means UTC; default now")
    hours: float = Field(24.0, gt=0, le=MAX_HOURS)
    min_elevation: float = Field(10.0, ge=0, le=89)

    @field_validator("tz")
    @classmethod
    def known_time_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"unknown time zone {value!r}")
        return value

    @field_validator("start")
    @classmethod
    def assume_utc(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


Params = Annotated[PassQuery, Query()]


# --- the prediction itself, shared by the JSON API, the HTML page and the map ---

@dataclass(frozen=True)
class Prediction:
    satellite: EarthSatellite
    station: GroundStation
    start: Time
    end: Time
    current: LookAngles
    below: SubPoint
    passes: list[Pass]


def predict(params: PassQuery, load_satellites_for: Callable[[str], list[EarthSatellite]]) -> Prediction:
    """Look up the satellite and run the same pass prediction the CLI does.

    Raises LookupError (with name suggestions) if the satellite isn't in the group.
    """
    satellite = lookup_satellite(load_satellites_for(params.group), params.sat)
    station = GroundStation(params.station, params.lat, params.lon, params.elev,
                            min_elevation_deg=params.min_elevation, timezone=params.tz)
    start = ts.from_datetime(params.start) if params.start else ts.now()
    return Prediction(
        satellite=satellite,
        station=station,
        start=start,
        end=start + timedelta(hours=params.hours),
        current=look_angles(satellite, station, start),
        below=sub_point(satellite, start),
        passes=find_passes(satellite, station, start, params.hours),
    )


def prediction_json(p: Prediction) -> dict:
    """Plain JSON: UTC ISO 8601 times, angles in degrees, distances in km."""
    def iso(moment: datetime) -> str:
        return moment.isoformat().replace("+00:00", "Z")

    return {
        "satellite": {
            "name": p.satellite.name.strip(),
            "norad_id": p.satellite.model.satnum,
            "tle_epoch": iso(p.satellite.epoch.utc_datetime()),
            "tle_age_days": round(tle_age_days(p.satellite, p.start), 3),
        },
        "station": {
            "name": p.station.name,
            "latitude_deg": p.station.latitude_deg,
            "longitude_deg": p.station.longitude_deg,
            "elevation_m": p.station.elevation_m,
            "min_elevation_deg": p.station.min_elevation_deg,
            "timezone": p.station.timezone,
        },
        "window": {"start": iso(p.start.utc_datetime()), "end": iso(p.end.utc_datetime())},
        "current": {
            "altitude_deg": round(p.current.altitude_deg, 3),
            "azimuth_deg": round(p.current.azimuth_deg, 3),
            "range_km": round(p.current.range_km, 1),
            "sub_point": {
                "latitude_deg": round(p.below.latitude_deg, 4),
                "longitude_deg": round(p.below.longitude_deg, 4),
                "height_km": round(p.below.height_km, 1),
            },
        },
        "passes": [
            {
                "rise": iso(x.rise),
                "culmination": iso(x.culmination),
                "set": iso(x.set),
                "max_elevation_deg": round(x.max_elevation_deg, 2),
                "rise_azimuth_deg": round(x.rise_azimuth_deg, 1),
                "set_azimuth_deg": round(x.set_azimuth_deg, 1),
                "duration_seconds": round(x.duration.total_seconds(), 1),
            }
            for x in p.passes
        ],
    }


# --- formatting for the HTML page (times shown in the station's time zone) ---

def format_duration(span: timedelta) -> str:
    """"6m 14s" for LEO passes; "6h 33m" for the hours-long passes of high orbits like GPS."""
    if span >= timedelta(hours=1):
        return hours_and_minutes(span)
    minutes, seconds = divmod(round(span.total_seconds()), 60)
    return f"{minutes}m {seconds:02d}s"


def hours_and_minutes(span: timedelta) -> str:
    hours, minutes = divmod(round(span.total_seconds() / 60), 60)
    return f"{hours}h {minutes:02d}m"


# Sky plot geometry (SVG user units). The plot is a view looking straight up: the center is
# the zenith (90 deg elevation), the outer circle is the horizon, north is at the top.
SKY_SIZE = 320
SKY_CENTER = SKY_SIZE / 2
SKY_RADIUS = 136


def sky_xy(azimuth_deg: float, altitude_deg: float) -> tuple[float, float]:
    """Polar projection: distance from the center grows linearly as elevation drops to 0."""
    r = SKY_RADIUS * (90 - max(altitude_deg, 0.0)) / 90
    a = math.radians(azimuth_deg)
    return round(SKY_CENTER + r * math.sin(a), 1), round(SKY_CENTER - r * math.cos(a), 1)


def sky_plot(p: Prediction) -> dict:
    """SVG-ready geometry for each pass's path across the sky, plus the elevation rings."""
    paths = []
    for number, x in enumerate(p.passes, start=1):
        track = sky_track(p.satellite, p.station, x)
        points = [sky_xy(az, el) for az, el in track]
        paths.append({
            "number": number,
            "d": "M" + " L".join(f"{px},{py}" for px, py in points),
            "start": points[0],
            "end": points[-1],
            "peak": sky_xy(*max(track, key=lambda sample: sample[1])),
        })
    return {
        "size": SKY_SIZE,
        "c": SKY_CENTER,
        "R": SKY_RADIUS,
        "mask_r": round(SKY_RADIUS * (90 - p.station.min_elevation_deg) / 90, 1),
        "rings": [{"r": round(SKY_RADIUS * (90 - el) / 90, 1), "label": f"{el}°"} for el in (30, 60)],
        "paths": paths,
    }


def coordinate(value: float, positive: str, negative: str, decimals: int = 4) -> str:
    return f"{abs(value):.{decimals}f}° {positive if value >= 0 else negative}"


LIVE_STEP_SECONDS = 30


def live_samples(p: Prediction, step_seconds: float = LIVE_STEP_SECONDS) -> dict:
    """SGP4 samples across the window, so the browser can show the predicted position *now*.

    Sample i is at min(i * step_s, duration_s) seconds after the start (evenly spaced, so the
    time isn't stored): [sub-point lat, lon, height km, east km, north km, up km].
    East/north/up is the satellite's position relative to the station. The browser interpolates
    that 3D vector (not the angles) between samples and converts it to azimuth/elevation:
    near the zenith the azimuth can swing through 180 degrees in seconds, so interpolating
    angles directly would give nonsense there, while the 3D position changes smoothly.
    """
    duration = (p.end - p.start) * 86400
    offsets = np.append(np.arange(0.0, duration, step_seconds), duration)
    times = p.start + offsets / 86400  # one Time holding every sample, so SGP4 runs vectorized

    altitude, azimuth, distance = (p.satellite - p.station.position()).at(times).altaz()
    alt, az, km = altitude.radians, azimuth.radians, distance.km
    east, north, up = km * np.cos(alt) * np.sin(az), km * np.cos(alt) * np.cos(az), km * np.sin(alt)

    geocentric = p.satellite.at(times)
    latitude, longitude = wgs84.latlon_of(geocentric)
    height = wgs84.height_of(geocentric).km

    def ms(moment: datetime) -> int:
        return round(moment.timestamp() * 1000)

    return {
        "start_ms": ms(p.start.utc_datetime()),
        "step_s": step_seconds,
        "duration_s": round(float(duration), 3),
        "mask_deg": p.station.min_elevation_deg,
        "sky": {"c": SKY_CENTER, "R": SKY_RADIUS},
        "passes": [[ms(x.rise), ms(x.set)] for x in p.passes],
        "samples": [
            [round(float(la), 3), round(float(lo), 3), round(float(h), 1),
             round(float(e), 1), round(float(n), 1), round(float(u), 1)]
            for la, lo, h, e, n, u in zip(latitude.degrees, longitude.degrees, height, east, north, up)
        ],
    }


def page_result(p: Prediction) -> dict:
    """Everything the template shows, already formatted, so the template stays simple."""
    zone = p.station.local_zone()
    start = p.start.utc_datetime()

    def local(moment: datetime, pattern: str = "%Y-%m-%d %H:%M %Z") -> str:
        return moment.astimezone(zone).strftime(pattern)

    if p.current.altitude_deg < 0:
        status = "below the horizon"
    elif p.current.altitude_deg < p.station.min_elevation_deg:
        status = "above the horizon, below the mask"
    else:
        status = "visible above the mask"

    next_pass = None
    if p.passes:
        first = p.passes[0]
        next_pass = {
            "rise": local(first.rise),
            "rise_time": local(first.rise, "%H:%M"),
            "rise_day": local(first.rise, "%a %d %b, %Z"),
            "direction": f"{compass_direction(first.rise_azimuth_deg)} → {compass_direction(first.set_azimuth_deg)}",
            "until": hours_and_minutes(first.rise - start),
            "max_el": f"{first.max_elevation_deg:.1f}°",
            "duration": format_duration(first.duration),
        }

    return {
        "name": p.satellite.name.strip(),
        "norad_id": p.satellite.model.satnum,
        "tle_epoch": p.satellite.epoch.utc_strftime("%Y-%m-%d %H:%M UTC"),
        "station_lat": coordinate(p.station.latitude_deg, "N", "S"),
        "station_lon": coordinate(p.station.longitude_deg, "E", "W"),
        "sky": sky_plot(p),
        "live": live_samples(p),
        "tle_age": f"{tle_age_days(p.satellite, p.start):.2f} days",
        "altitude": f"{p.current.altitude_deg:.1f}°",
        "azimuth": f"{p.current.azimuth_deg:.0f}° ({compass_direction(p.current.azimuth_deg)})",
        "range_km": f"{p.current.range_km:,.0f}",
        "visible": p.current.altitude_deg >= p.station.min_elevation_deg,
        "status": status,
        "sub_point": f"{coordinate(p.below.latitude_deg, 'N', 'S', 1)} {coordinate(p.below.longitude_deg, 'E', 'W', 1)}",
        "height_km": f"{p.below.height_km:.0f}",
        "next_pass": next_pass,
        "window_start": local(start),
        "window_end": local(p.end.utc_datetime()),
        "passes": [
            {
                "date": local(x.rise, "%Y-%m-%d"),
                "rise": local(x.rise, "%H:%M:%S"),
                "peak": local(x.culmination, "%H:%M:%S"),
                "set": local(x.set, "%H:%M:%S"),
                "max_el": f"{x.max_elevation_deg:.1f}°",
                "direction": f"{compass_direction(x.rise_azimuth_deg)} → {compass_direction(x.set_azimuth_deg)}",
                "duration": format_duration(x.duration),
            }
            for x in p.passes
        ],
    }


# --- routes ---

@app.get("/", response_class=HTMLResponse)
def index(request: Request, params: Params, load_satellites_for: GroupLoader) -> HTMLResponse:
    """The form, plus results for whatever it was submitted with (ISS over West Lafayette by default)."""
    result, error, names, map_query, status = None, None, [], "", 200
    try:
        names = sorted(s.name.strip() for s in load_satellites_for(params.group))
        prediction = predict(params, load_satellites_for)
        result = page_result(prediction)
        # Pin the map and JSON links to the same start time, so they match the table exactly.
        fixed_start = prediction.start.utc_datetime().isoformat()
        map_query = urlencode({**params.model_dump(exclude_none=True), "start": fixed_start})
    except LookupError as lookup_error:
        error, status = str(lookup_error), 404
    except OSError as download_error:
        error, status = f"{DOWNLOAD_FAILED} (Reason: {download_error})", 503

    context = {"params": params, "groups": GROUP_LABELS, "max_hours": MAX_HOURS, "satellite_names": names,
               "result": result, "error": error, "map_query": map_query}
    return templates.TemplateResponse(request, "index.html", context, status_code=status)


# --- per-pass detail page ---

class PassDetailQuery(PassQuery):
    """The overview's parameters plus the radio frequency for the Doppler curve.

    It's one model (not a separate query argument) because FastAPI only expands a Pydantic model
    into query parameters when it's the route's sole query input.
    """

    freq_mhz: float = Field(ISS_DOWNLINK_HZ / 1e6, gt=1, le=30000,
                            description="Downlink frequency in MHz (default: the ISS's 145.800 MHz voice downlink)")


DetailParams = Annotated[PassDetailQuery, Query()]
PassNumber = Annotated[int, PathParam(ge=1, le=500, description="Pass number, 1 = the first pass in the window")]

CHART_WIDTH, CHART_HEIGHT = 640, 150


def chart_path(xs: list[float], ys: list[float], y_min: float, y_max: float) -> str:
    """An SVG polyline path for (x, y) data, with x spanning the full chart width."""
    x_span = (xs[-1] - xs[0]) or 1.0

    def point(x: float, y: float) -> str:
        px = (x - xs[0]) / x_span * CHART_WIDTH
        py = CHART_HEIGHT - (y - y_min) / (y_max - y_min) * CHART_HEIGHT
        return f"{px:.1f},{py:.1f}"

    return "M" + " L".join(point(x, y) for x, y in zip(xs, ys))


def select_pass(p: Prediction, number: int) -> Pass:
    if number > len(p.passes):
        raise HTTPException(status_code=404, detail=f"there is no pass {number}; this window has {len(p.passes)}")
    return p.passes[number - 1]


def pass_page_result(p: Prediction, number: int, profile: PassProfile) -> dict:
    """Everything the pass page shows, formatted, plus SVG geometry for its charts."""
    zone = p.station.local_zone()
    x = select_pass(p, number)

    def local(moment: datetime, pattern: str = "%H:%M:%S") -> str:
        return moment.astimezone(zone).strftime(pattern)

    seconds = [(t - x.rise).total_seconds() for t in profile.times]
    doppler_khz = (profile.doppler_hz / 1000).tolist()
    swing = max(abs(min(doppler_khz)), abs(max(doppler_khz)))
    limit = max(0.5, math.ceil(swing * 2) / 2)  # symmetric axis, rounded up to 0.5 kHz
    peak_rate = float(profile.azimuth_rate_deg_s.max())

    track = sky_track(p.satellite, p.station, x)
    sky_points = [sky_xy(az, el) for az, el in track]

    every_30s = [i for i, s in enumerate(seconds) if int(s) % 30 == 0] + [len(seconds) - 1]
    return {
        "number": number,
        "count": len(p.passes),
        "name": p.satellite.name.strip(),
        "norad_id": p.satellite.model.satnum,
        "date": local(x.rise, "%A %d %B %Y"),
        "zone": local(x.rise, "%Z"),
        "rise": local(x.rise), "peak": local(x.culmination), "set": local(x.set),
        "max_el": f"{x.max_elevation_deg:.1f}°",
        "duration": format_duration(x.duration),
        "direction": f"{compass_direction(x.rise_azimuth_deg)} → {compass_direction(x.set_azimuth_deg)}",
        "frequency_mhz": f"{profile.frequency_hz / 1e6:.3f}",
        "doppler_start": f"{profile.doppler_hz[0] / 1000:+.2f} kHz",
        "doppler_end": f"{profile.doppler_hz[-1] / 1000:+.2f} kHz",
        "tune_start": f"{profile.received_hz[0] / 1e6:.4f}",
        "tune_end": f"{profile.received_hz[-1] / 1e6:.4f}",
        "min_range": f"{profile.range_km.min():,.0f}",
        "peak_az_rate": f"{peak_rate:.1f}",
        "keyhole": peak_rate > 5,
        "charts": {
            "w": CHART_WIDTH, "h": CHART_HEIGHT,
            "doppler": chart_path(seconds, doppler_khz, -limit, limit),
            "doppler_limit": f"{limit:g}",
            "elevation": chart_path(seconds, profile.elevation_deg.tolist(), 0, 90),
            "mask_y": round(CHART_HEIGHT - p.station.min_elevation_deg / 90 * CHART_HEIGHT, 1),
            "peak_x": round((x.culmination - x.rise).total_seconds() / (seconds[-1] or 1) * CHART_WIDTH, 1),
            "rise": local(x.rise, "%H:%M"), "peak": local(x.culmination, "%H:%M:%S"), "set": local(x.set, "%H:%M"),
        },
        "sky": {
            "size": SKY_SIZE, "c": SKY_CENTER, "R": SKY_RADIUS,
            "mask_r": round(SKY_RADIUS * (90 - p.station.min_elevation_deg) / 90, 1),
            "rings": [{"r": round(SKY_RADIUS * (90 - el) / 90, 1), "label": f"{el}°"} for el in (30, 60)],
            "d": "M" + " L".join(f"{px},{py}" for px, py in sky_points),
            "start": sky_points[0], "end": sky_points[-1],
        },
        "rows": [
            {
                "time": local(profile.times[i]),
                "az": f"{profile.azimuth_deg[i]:.1f}°",
                "el": f"{profile.elevation_deg[i]:.1f}°",
                "range": f"{profile.range_km[i]:,.0f}",
                "doppler": f"{profile.doppler_hz[i]:+,.0f}",
                "tune": f"{profile.received_hz[i] / 1e6:.4f}",
            }
            for i in sorted(set(every_30s))
        ],
    }


def _pass_and_profile(params: PassDetailQuery, load_satellites_for: Callable[[str], list[EarthSatellite]], number: int):
    p = _predict_or_http_error(params, load_satellites_for)
    x = select_pass(p, number)
    return p, pass_profile(p.satellite, p.station, x, frequency_hz=params.freq_mhz * 1e6)


@app.get("/pass/{number}", response_class=HTMLResponse)
def pass_page(request: Request, number: PassNumber, params: DetailParams, load_satellites_for: GroupLoader) -> HTMLResponse:
    """One pass in detail: Doppler curve, elevation profile, sky track and pointing table."""
    p, profile = _pass_and_profile(params, load_satellites_for, number)
    freq_mhz = params.freq_mhz
    # Pin the start time so links back to the overview and between passes show the same window.
    query = {**params.model_dump(exclude_none=True, exclude={"freq_mhz"}), "start": p.start.utc_datetime().isoformat()}
    context = {
        "params": params,
        "result": {"station_lat": coordinate(p.station.latitude_deg, "N", "S"),
                   "station_lon": coordinate(p.station.longitude_deg, "E", "W")},
        "detail": pass_page_result(p, number, profile),
        "map_query": urlencode(query),
        "freq_query": urlencode({**query, "freq_mhz": freq_mhz}),
        "freq_mhz": freq_mhz,
        "hidden_params": query,
    }
    return templates.TemplateResponse(request, "pass.html", context)


@app.get("/pass/{number}/pointing.csv")
def pass_pointing_csv(number: PassNumber, params: DetailParams, load_satellites_for: GroupLoader) -> Response:
    """The pass sampled every second: azimuth, elevation, range, range rate, Doppler, tuning frequency."""
    p, profile = _pass_and_profile(params, load_satellites_for, number)
    filename = f"orbittrack_{p.satellite.model.satnum}_pass{number:02d}_{profile.times[0]:%Y%m%dT%H%M%S}Z.csv"
    return Response(pointing_csv(profile), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# --- accuracy page: our SGP4 predictions vs NASA's ISS ephemeris ---

ACCURACY_CHART_W, ACCURACY_CHART_H = 640, 170


def accuracy_page_result(satellite: EarthSatellite, ephemeris: Ephemeris) -> dict:
    """The live comparison, formatted, with SVG paths for the divergence charts."""
    result = compare(satellite, ephemeris)
    start = ephemeris.times[0]
    days = np.array([(t - start).total_seconds() / 86400 for t in ephemeris.times])
    keep = np.arange(0, len(days), 8)  # every 32 minutes is plenty for a chart

    def path(values: np.ndarray, low: float, high: float) -> str:
        span_days = days[-1] or 1.0
        points = (f"{d / span_days * ACCURACY_CHART_W:.1f},"
                  f"{ACCURACY_CHART_H - (v - low) / (high - low) * ACCURACY_CHART_H:.1f}"
                  for d, v in zip(days[keep], values[keep]))
        return "M" + " L".join(points)

    total_top = max(5.0, float(np.ceil(result.total_km.max() / 10) * 10))
    side = max(1.0, float(np.ceil(max(np.abs(result.cross_km).max(), np.abs(result.radial_km).max()))))

    # Cross-track and radial differences oscillate once per orbit (~92 min); drawn point by point
    # that's an unreadable wall of zigzags. The meaningful quantity is how big the swing gets,
    # so draw its envelope: the largest absolute value in each 3-hour window.
    windows = np.floor(days * 8).astype(int)  # 8 windows per day
    edges = [(days[windows == w].mean(), np.abs(result.cross_km[windows == w]).max(),
              np.abs(result.radial_km[windows == w]).max()) for w in np.unique(windows)]
    env_days, env_cross, env_radial = (np.array(column) for column in zip(*edges))

    def envelope_path(values: np.ndarray) -> str:
        span_days = days[-1] or 1.0
        return "M" + " L".join(f"{d / span_days * ACCURACY_CHART_W:.1f},{ACCURACY_CHART_H - v / side * ACCURACY_CHART_H:.1f}"
                               for d, v in zip(env_days, values))
    age_hours = float(result.tle_age_days[0]) * 24
    return {
        "tle_epoch": satellite.epoch.utc_strftime("%Y-%m-%d %H:%M UTC"),
        "oem_start": f"{start:%Y-%m-%d %H:%M} UTC",
        "oem_end": f"{ephemeris.times[-1]:%Y-%m-%d %H:%M} UTC",
        "oem_count": f"{len(ephemeris.times):,}",
        "start_km": f"{result.total_km[0]:.2f}",
        "start_timing": f"{result.timing_error_s[0]:+.2f}",
        "tle_age": (f"{age_hours / 24:.2f} days old" if age_hours >= 0
                    else f"{-age_hours:.1f} h newer than NASA's file start"),
        "along_share": f"{np.sum(result.along_km ** 2) / np.sum(result.total_km ** 2):.1%}",
        "max_cross": f"{np.abs(result.cross_km).max():.1f}",
        "max_radial": f"{np.abs(result.radial_km).max():.1f}",
        "rows": [
            {
                "day": round((row["time"] - start).total_seconds() / 86400),
                "time": f"{row['time']:%d %b %H:%M}",
                "total": f"{row['total_km']:.2f}",
                "along": f"{row['along_km']:+.2f}",
                "cross": f"{row['cross_km']:+.2f}",
                "radial": f"{row['radial_km']:+.2f}",
                "timing": f"{row['timing_s']:+.2f}",
            }
            for row in summary_rows(result)
        ],
        "chart": {
            "w": ACCURACY_CHART_W, "h": ACCURACY_CHART_H, "days": f"{days[-1]:.0f}",
            "total": path(result.total_km, 0, total_top), "total_top": f"{total_top:g}",
            "total_mid": f"{total_top / 2:g}",
            "cross": envelope_path(env_cross), "radial": envelope_path(env_radial),
            "side": f"{side:g}", "side_mid": f"{side / 2:g}",
        },
    }


@app.get("/accuracy", response_class=HTMLResponse)
def accuracy_page(request: Request, load_satellites_for: GroupLoader, load_ephemeris: EphemerisLoader) -> HTMLResponse:
    """How close our SGP4/TLE predictions are to NASA's own ISS trajectory, checked live."""
    live, error = None, None
    try:
        iss = lookup_satellite(load_satellites_for("stations"), "25544")
        live = accuracy_page_result(iss, load_ephemeris())
    except (OSError, LookupError) as problem:
        error = str(problem)
    # The masthead belongs to the overview; this page has its own, so give it neutral defaults.
    context = {"params": PassQuery(), "result": None, "live": live, "error": error, "map_query": ""}
    return templates.TemplateResponse(request, "accuracy.html", context)


@app.get("/map", response_class=HTMLResponse)
def map_page(params: Params, load_satellites_for: GroupLoader) -> HTMLResponse:
    """The interactive ground-track map for the same parameters (embedded by the main page)."""
    p = _predict_or_http_error(params, load_satellites_for)
    return HTMLResponse(build_map(p.satellite, p.station, p.start, p.end, p.passes).get_root().render())


@app.get("/api/passes")
def api_passes(params: Params, load_satellites_for: GroupLoader) -> dict:
    """Current position and every pass above the mask in the window, as JSON."""
    return prediction_json(_predict_or_http_error(params, load_satellites_for))


@app.get("/healthz")
def health() -> dict:
    """For the hosting platform's health checks: answers instantly, without CelesTrak or orbit math."""
    return {"status": "ok"}


def _predict_or_http_error(params: PassQuery, load_satellites_for: Callable[[str], list[EarthSatellite]]) -> Prediction:
    try:
        return predict(params, load_satellites_for)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except OSError as error:
        raise HTTPException(status_code=503, detail=f"{DOWNLOAD_FAILED} (Reason: {error})")
