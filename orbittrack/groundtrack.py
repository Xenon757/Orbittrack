"""Ground tracks: the path traced on Earth by the point directly beneath a satellite."""

from datetime import datetime
from typing import NamedTuple

import numpy as np
from skyfield.api import EarthSatellite, wgs84
from skyfield.timelib import Time

SECONDS_PER_DAY = 86400.0


class TrackPoint(NamedTuple):
    time: datetime  # UTC
    latitude_deg: float
    longitude_deg: float  # -180..180
    height_km: float  # above the WGS84 ellipsoid


def ground_track(satellite: EarthSatellite, start: Time, end: Time, step_seconds: float = 30.0) -> list[TrackPoint]:
    """Sub-satellite points from `start` to `end`, one every `step_seconds`.

    The track drifts west on every orbit even though the orbit itself barely moves in space:
    the Earth rotates ~22.5 deg east underneath the ISS during each ~92-minute lap.
    """
    duration_seconds = (end - start) * SECONDS_PER_DAY
    offsets = np.append(np.arange(0.0, duration_seconds, step_seconds), duration_seconds)
    times = start + offsets / SECONDS_PER_DAY  # one Time holding every sample, so SGP4 runs vectorized

    geocentric = satellite.at(times)
    latitude, longitude = wgs84.latlon_of(geocentric)
    height = wgs84.height_of(geocentric)

    return [
        TrackPoint(moment, float(lat), float(lon), float(h))
        for moment, lat, lon, h in zip(times.utc_datetime(), latitude.degrees, longitude.degrees, height.km)
    ]


def split_at_antimeridian(points: list[TrackPoint]) -> list[list[TrackPoint]]:
    """Break the track wherever it crosses longitude +/-180.

    Longitude wraps from +180 to -180 there, so drawing one continuous line would put a
    horizontal streak across the whole map. Consecutive samples are only seconds apart,
    so a jump of more than 180 deg can only mean we wrapped around.
    """
    segments: list[list[TrackPoint]] = []
    for point in points:
        if segments and abs(point.longitude_deg - segments[-1][-1].longitude_deg) <= 180:
            segments[-1].append(point)
        else:
            segments.append([point])
    return segments
