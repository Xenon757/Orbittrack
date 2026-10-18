"""Pass prediction tests, using the fixed 2026-10-18 TLE and a fixed start time (offline, deterministic)."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from skyfield.api import load

from orbittrack.observe import look_angles
from orbittrack.passes import CULMINATE, RISE, SET, compass_direction, find_passes, group_into_passes
from orbittrack.station import WEST_LAFAYETTE
from orbittrack.tle import find_satellite, load_satellites

FIXTURE = Path(__file__).parent / "fixtures" / "stations_2026-10-18.tle"
ts = load.timescale()
START = ts.utc(2026, 10, 18, 12, 0, 0)  # about an hour after the fixture's TLE epoch
HOURS = 24.0


@pytest.fixture(scope="module")
def iss():
    return find_satellite(load_satellites(FIXTURE), "25544")


@pytest.fixture(scope="module")
def passes(iss):
    return find_passes(iss, WEST_LAFAYETTE, START, HOURS)


# --- grouping raw events into passes (pure logic, no orbit math) ---

def test_groups_complete_passes():
    events = [RISE, CULMINATE, SET, RISE, CULMINATE, SET]
    assert group_into_passes(events) == [(0, [1], 2), (3, [4], 5)]


def test_drops_pass_already_in_progress_at_window_start():
    events = [CULMINATE, SET, RISE, CULMINATE, SET]
    assert group_into_passes(events) == [(2, [3], 4)]


def test_drops_pass_still_in_progress_at_window_end():
    events = [RISE, CULMINATE, SET, RISE, CULMINATE]
    assert group_into_passes(events) == [(0, [1], 2)]


def test_keeps_every_culmination_in_a_pass():
    assert group_into_passes([RISE, CULMINATE, CULMINATE, SET]) == [(0, [1, 2], 3)]


def test_no_events_means_no_passes():
    assert group_into_passes([]) == []


# --- predicted passes for the real ISS TLE ---

def test_finds_the_expected_passes(passes):
    # Regression check against this fixed TLE + start time; the ISS typically gets
    # 4-6 passes a day above 10 deg at Purdue's latitude.
    assert len(passes) == 6


def test_rise_before_peak_before_set(passes):
    for p in passes:
        assert p.rise < p.culmination < p.set


def test_passes_are_in_order_and_inside_the_window(passes):
    window_start = START.utc_datetime()
    window_end = window_start + timedelta(hours=HOURS)
    for earlier, later in zip(passes, passes[1:]):
        assert earlier.set < later.rise
    assert window_start <= passes[0].rise
    assert passes[-1].set <= window_end


def test_max_elevation_is_above_the_mask(passes):
    for p in passes:
        assert WEST_LAFAYETTE.min_elevation_deg <= p.max_elevation_deg <= 90


def test_rise_and_set_times_are_accurate_to_a_second(iss, passes):
    # find_events() root-finds each mask crossing to within half a second, so one second
    # before a rise the satellite must still be below the mask and one second after it must
    # be above (and the reverse for a set). This proves the times aren't just rough samples.
    mask = WEST_LAFAYETTE.min_elevation_deg
    second = timedelta(seconds=1)

    def altitude(moment):
        return look_angles(iss, WEST_LAFAYETTE, ts.from_datetime(moment)).altitude_deg

    for p in passes:
        assert altitude(p.rise - second) < mask < altitude(p.rise + second)
        assert altitude(p.set - second) > mask > altitude(p.set + second)


def test_durations_are_sane(passes):
    # An ISS pass above 10 deg can't last more than about 8 minutes: at ~430 km it
    # crosses the visible patch of sky quickly.
    for p in passes:
        assert timedelta(seconds=30) < p.duration < timedelta(minutes=10)


def test_higher_mask_gives_fewer_and_shorter_passes(iss, passes):
    strict_station = replace(WEST_LAFAYETTE, min_elevation_deg=30.0)
    strict = find_passes(iss, strict_station, START, HOURS)

    assert len(strict) < len(passes)
    assert all(p.max_elevation_deg >= 30.0 for p in strict)
    for s in strict:
        # Each strict pass sits inside the matching 10-degree pass.
        matching = next(p for p in passes if p.rise < s.culmination < p.set)
        assert s.duration < matching.duration


# --- compass labels ---

@pytest.mark.parametrize("azimuth, label", [(0, "N"), (359, "N"), (44, "NE"), (90, "E"), (225, "SW"), (300, "NW")])
def test_compass_direction(azimuth, label):
    assert compass_direction(azimuth) == label
