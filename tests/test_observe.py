"""Look-angle geometry checks using a fixed TLE and a fixed time (deterministic, offline).

Rather than comparing against magic numbers, these use geometric facts that must hold:
a station directly under the satellite sees it straight overhead, and a station on the
opposite side of the Earth sees it straight below.
"""

import pytest
from skyfield.api import load

from orbittrack.observe import look_angles, sub_point
from orbittrack.station import WEST_LAFAYETTE, GroundStation


@pytest.fixture
def t():
    # A few hours after the fixture's TLE epoch (2026-10-18 11:12 UTC).
    return load.timescale().utc(2026, 10, 18, 15, 0, 0)


def test_iss_height_is_low_earth_orbit(iss, t):
    assert 380 < sub_point(iss, t).height_km < 460


def test_station_under_satellite_sees_it_overhead(iss, t):
    below = sub_point(iss, t)
    station = GroundStation("sub-point", below.latitude_deg, below.longitude_deg, elevation_m=0.0)

    angles = look_angles(iss, station, t)

    assert angles.altitude_deg == pytest.approx(90.0, abs=0.01)
    assert angles.range_km == pytest.approx(below.height_km, abs=0.1)


def test_station_on_far_side_of_earth_sees_it_below(iss, t):
    below = sub_point(iss, t)
    antipode_lon = below.longitude_deg + 180 if below.longitude_deg < 0 else below.longitude_deg - 180
    station = GroundStation("antipode", -below.latitude_deg, antipode_lon, elevation_m=0.0)

    assert look_angles(iss, station, t).altitude_deg == pytest.approx(-90.0, abs=0.5)


def test_angles_are_in_valid_ranges(iss, t):
    angles = look_angles(iss, WEST_LAFAYETTE, t)
    assert -90 <= angles.altitude_deg <= 90
    assert 0 <= angles.azimuth_deg < 360
    assert angles.range_km > 0
