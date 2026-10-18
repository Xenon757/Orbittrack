"""Pass prediction: when a satellite rises above, peaks over, and sets below the elevation mask."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from skyfield.api import EarthSatellite, load
from skyfield.timelib import Time

from orbittrack.observe import look_angles
from orbittrack.station import GroundStation

_ts = load.timescale()

# Event codes returned by Skyfield's find_events().
RISE, CULMINATE, SET = 0, 1, 2


@dataclass(frozen=True)
class Pass:
    rise: datetime  # UTC, when the satellite climbs above the mask
    culmination: datetime  # UTC, when it is highest in the sky
    set: datetime  # UTC, when it drops below the mask
    max_elevation_deg: float
    rise_azimuth_deg: float
    set_azimuth_deg: float

    @property
    def duration(self) -> timedelta:
        """Time above the mask: the usable contact window for the ground station."""
        return self.set - self.rise


def group_into_passes(events: list[int]) -> list[tuple[int, list[int], int]]:
    """Group a flat list of event codes into complete passes.

    Returns (rise_index, culmination_indices, set_index) for each pass. A pass that is
    already in progress when the search window opens (no rise) or still in progress when
    it closes (no set) is dropped, because we can't report its true rise/set times.
    A pass can have more than one culmination (rare, for odd geometries), so we keep them all.
    """
    passes = []
    rise = None
    culminations: list[int] = []

    for i, event in enumerate(events):
        if event == RISE:
            rise, culminations = i, []
        elif event == CULMINATE and rise is not None:
            culminations.append(i)
        elif event == SET:
            if rise is not None and culminations:
                passes.append((rise, culminations, i))
            rise, culminations = None, []

    return passes


def find_passes(
    satellite: EarthSatellite, station: GroundStation, start: Time, hours: float = 24.0
) -> list[Pass]:
    """Predict every complete pass above the station's elevation mask in [start, start + hours].

    find_events() samples the satellite's altitude angle over the window and then
    root-finds the exact moments it crosses the mask (rise/set) and peaks (culmination),
    which is much more precise than stepping through time minute by minute.
    """
    end = start + timedelta(hours=hours)
    times, events = satellite.find_events(
        station.position(), start, end, altitude_degrees=station.min_elevation_deg
    )

    passes = []
    for rise_i, culmination_is, set_i in group_into_passes(list(events)):
        peak_i = max(culmination_is, key=lambda i: look_angles(satellite, station, times[i]).altitude_deg)
        passes.append(
            Pass(
                rise=times[rise_i].utc_datetime(),
                culmination=times[peak_i].utc_datetime(),
                set=times[set_i].utc_datetime(),
                max_elevation_deg=look_angles(satellite, station, times[peak_i]).altitude_deg,
                rise_azimuth_deg=look_angles(satellite, station, times[rise_i]).azimuth_deg,
                set_azimuth_deg=look_angles(satellite, station, times[set_i]).azimuth_deg,
            )
        )
    return passes


def sky_track(satellite: EarthSatellite, station: GroundStation, p: Pass, points: int = 48) -> list[tuple[float, float]]:
    """(azimuth, altitude) in degrees from rise to set: evenly spaced samples plus the exact peak.

    This is the path the satellite traces across the station's sky: what an antenna would
    have to follow, and what a polar "sky plot" draws.
    """
    start = _ts.from_datetime(p.rise)
    days = p.duration.total_seconds() / 86400
    peak = (p.culmination - p.rise).total_seconds() / 86400
    # Even spacing plus the exact peak: near the zenith the altitude changes so fast that even
    # spacing alone can miss the top of a high pass by a degree or more.
    offsets = np.sort(np.append(np.linspace(0.0, days, points), peak))
    times = start + offsets  # one Time holding every sample, so SGP4 runs vectorized
    altitude, azimuth, _ = (satellite - station.position()).at(times).altaz()
    return list(zip(azimuth.degrees.tolist(), altitude.degrees.tolist()))


def compass_direction(azimuth_deg: float) -> str:
    """Turn an azimuth into an 8-point compass label (e.g. 315 -> "NW") for human-readable output."""
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return labels[round(azimuth_deg / 45) % 8]
