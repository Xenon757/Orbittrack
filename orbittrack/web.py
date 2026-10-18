"""Web interface: the same predictions as the CLI, served over HTTP.

Run locally with:
    uvicorn orbittrack.web:app --reload
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal, get_args
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from skyfield.api import EarthSatellite, load
from skyfield.timelib import Time

from orbittrack.observe import LookAngles, SubPoint, look_angles, sub_point
from orbittrack.passes import Pass, find_passes
from orbittrack.station import WEST_LAFAYETTE, GroundStation
from orbittrack.tle import fetch_tle_file, load_satellites, lookup_satellite, tle_age_days

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
ts = load.timescale()


# --- TLE data shared by every visitor ---

_cache_lock = threading.Lock()
_parsed: dict[str, tuple[float, list[EarthSatellite]]] = {}  # group -> (file mtime, satellites)


def load_group(group: str) -> list[EarthSatellite]:
    """Satellites for a CelesTrak group, downloaded at most every few hours and parsed once.

    FastAPI runs requests on several threads. The lock makes sure that when the cache goes
    stale, only one request downloads a fresh copy while the others wait and reuse it.
    """
    with _cache_lock:
        path = fetch_tle_file(group)
        modified = path.stat().st_mtime
        if group not in _parsed or _parsed[group][0] != modified:
            _parsed[group] = (modified, load_satellites(path))
        return _parsed[group][1]


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


# --- routes ---

@app.get("/api/passes")
def api_passes(params: Params, load_satellites_for: GroupLoader) -> dict:
    """Current position and every pass above the mask in the window, as JSON."""
    try:
        return prediction_json(predict(params, load_satellites_for))
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error))
