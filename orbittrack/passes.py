"""Pass prediction: when a satellite rises above, peaks over, and sets below the elevation mask."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from skyfield.api import EarthSatellite
from skyfield.timelib import Time

from orbittrack.observe import look_angles
from orbittrack.station import GroundStation

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


def compass_direction(azimuth_deg: float) -> str:
    """Turn an azimuth into an 8-point compass label (e.g. 315 -> "NW") for human-readable output."""
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return labels[round(azimuth_deg / 45) % 8]
