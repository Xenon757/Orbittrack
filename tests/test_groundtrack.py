"""Ground track and coverage geometry, using the fixed 2026-10-18 TLE (offline, deterministic)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from skyfield.api import load

from orbittrack.groundtrack import (
    EARTH_RADIUS_KM, TrackPoint, circle_around, coverage_radius_deg, ground_track, split_at_antimeridian,
)
from orbittrack.observe import look_angles, sub_point
from orbittrack.station import GroundStation

ts = load.timescale()
START = ts.utc(2026, 10, 18, 12, 0, 0)
END = START + timedelta(hours=24)


def great_circle_deg(lat1, lon1, lat2, lon2):
    """Angle at Earth's center between two points (haversine formula), in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(a)))


@pytest.fixture(scope="module")
def track(iss):
    return ground_track(iss, START, END)


def point(longitude):
    return TrackPoint(datetime(2026, 1, 1, tzinfo=timezone.utc), 0.0, longitude, 400.0)


# --- ground track ---

def test_track_covers_the_whole_window_at_the_requested_step(track):
    assert track[0].time == START.utc_datetime()
    assert abs(track[-1].time - END.utc_datetime()) < timedelta(milliseconds=1)
    assert track[1].time - track[0].time == pytest.approx(timedelta(seconds=30), abs=timedelta(milliseconds=1))


def test_track_latitude_is_bounded_by_the_orbit_inclination(iss, track):
    inclination = np.degrees(iss.model.inclo)  # 51.63 deg for the ISS
    highest = max(abs(p.latitude_deg) for p in track)
    # Geodetic (map) latitude can exceed the geocentric inclination by ~0.2 deg because
    # Earth is slightly flattened; over a full day the track should get right up to that limit.
    assert inclination - 0.5 < highest < inclination + 0.3


def test_ground_speed_is_about_seven_km_per_second(track):
    # 7.66 km/s orbital speed, scaled down to the surface (R / (R + h)) and reduced a little
    # by Earth's rotation in the same direction, gives ~6.9 km/s along the ground.
    lats = np.array([p.latitude_deg for p in track])
    lons = np.array([p.longitude_deg for p in track])
    km = np.radians(great_circle_deg(lats[:-1], lons[:-1], lats[1:], lons[1:])) * EARTH_RADIUS_KM
    assert np.all((km / 30 > 6.5) & (km / 30 < 7.3))


def test_track_splits_once_per_antimeridian_crossing(track):
    segments = split_at_antimeridian(track)
    crossings = sum(abs(b.longitude_deg - a.longitude_deg) > 180 for a, b in zip(track, track[1:]))
    assert len(segments) == crossings + 1
    assert sum(len(s) for s in segments) == len(track)  # no points lost or duplicated


def test_split_at_antimeridian_on_simple_cases():
    assert split_at_antimeridian([]) == []
    assert len(split_at_antimeridian([point(-10), point(0), point(10)])) == 1
    segments = split_at_antimeridian([point(170), point(179), point(-179), point(-170)])
    assert [[p.longitude_deg for p in s] for s in segments] == [[170, 179], [-179, -170]]


# --- coverage footprint ---

def test_coverage_radius_basic_geometry():
    # 0 deg mask: the horizon distance, where the line of sight just grazes the Earth.
    assert coverage_radius_deg(420, 0) == pytest.approx(np.degrees(np.arccos(EARTH_RADIUS_KM / (EARTH_RADIUS_KM + 420))))
    assert coverage_radius_deg(420, 10) < coverage_radius_deg(420, 0)  # higher mask, smaller circle
    assert coverage_radius_deg(800, 10) > coverage_radius_deg(420, 10)  # higher orbit, bigger circle
    assert coverage_radius_deg(420, 90) == pytest.approx(0, abs=1e-9)  # straight up only


def test_circle_points_are_all_the_same_distance_from_the_center():
    points = circle_around(40.4, -86.9, 12.5)
    assert points[0] == pytest.approx(points[-1])  # closed loop
    for lat, lon in points:
        assert great_circle_deg(40.4, -86.9, lat, lon) == pytest.approx(12.5, abs=1e-6)


@pytest.mark.parametrize("mask_deg", [0.0, 10.0, 30.0])
def test_station_on_the_coverage_edge_sees_the_satellite_at_the_mask(iss, mask_deg):
    # Place stations all around the coverage circle of the ISS's current ground point.
    # Each should see the ISS right at the mask; the small error is the sphere-vs-ellipsoid
    # approximation in the coverage formula.
    t = ts.utc(2026, 10, 18, 15)
    below = sub_point(iss, t)
    radius = coverage_radius_deg(below.height_km, mask_deg)
    for lat, lon in circle_around(below.latitude_deg, below.longitude_deg, radius, points=12):
        elevation = look_angles(iss, GroundStation("edge", lat, lon, 0.0), t).altitude_deg
        assert elevation == pytest.approx(mask_deg, abs=0.2)
