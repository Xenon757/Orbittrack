"""Pass prediction tests, using the fixed 2026-10-18 TLE and a fixed start time (offline, deterministic)."""

from dataclasses import replace
from datetime import timedelta

import pytest
from skyfield.api import load

from orbittrack.observe import look_angles
from orbittrack.passes import CULMINATE, RISE, SET, compass_direction, find_passes, group_into_passes
from orbittrack.station import WEST_LAFAYETTE

ts = load.timescale()
START = ts.utc(2026, 10, 18, 12, 0, 0)  # about an hour after the fixture's TLE epoch
HOURS = 24.0


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


def test_window_starting_mid_pass_drops_only_that_pass(iss, passes):
    # 16:23 UTC is in the middle of the 16:20-16:26 UTC pass. End at the same time as the
    # full window, so the result should be the full list minus that pass and the ones before it.
    mid_pass_start = ts.utc(2026, 10, 18, 16, 23)
    later = find_passes(iss, WEST_LAFAYETTE, mid_pass_start, HOURS - (4 + 23 / 60))

    in_progress = next(p for p in passes if p.rise < mid_pass_start.utc_datetime() < p.set)
    expected = [p for p in passes if p.rise > in_progress.set]

    # Same passes, but not bit-identical: find_events() is accurate to half a second, and a
    # different window start lands its search slightly differently (we saw ~9 ms).
    assert len(later) == len(expected)
    for got, want in zip(later, expected):
        assert abs(got.rise - want.rise) < timedelta(seconds=0.5)
        assert abs(got.set - want.set) < timedelta(seconds=0.5)


@pytest.mark.parametrize("latitude, sees_iss", [(60.0, True), (80.0, False)])
def test_iss_is_never_visible_far_beyond_its_inclination(iss, latitude, sees_iss):
    # The ISS orbit is inclined 51.6 deg, so its ground track never goes past 51.6 N/S.
    # From ~420 km it's above the horizon out to ~20 deg of arc, so it can't even rise
    # (0 deg mask) for a station beyond roughly 72 N. 60 N is inside that, 80 N is not.
    station = replace(WEST_LAFAYETTE, latitude_deg=latitude, min_elevation_deg=0.0)
    found = find_passes(iss, station, START, hours=72)
    assert (len(found) > 0) == sees_iss


# --- compass labels ---

@pytest.mark.parametrize("azimuth, label", [(0, "N"), (359, "N"), (44, "NE"), (90, "E"), (225, "SW"), (300, "NW")])
def test_compass_direction(azimuth, label):
    assert compass_direction(azimuth) == label
