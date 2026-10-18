"""Cross-check find_passes() against a completely different, deliberately simple method.

find_passes() uses Skyfield's find_events(), which samples coarsely and then root-finds
each crossing. Here we instead brute-force it: compute the altitude every couple of seconds
and treat each unbroken run of samples above the mask as a pass. If the clever method ever
skipped a short pass or mistimed a crossing, the two would disagree.
"""

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest
from skyfield.api import load

from orbittrack.passes import find_passes
from orbittrack.station import WEST_LAFAYETTE

ts = load.timescale()
START = (2026, 10, 18, 12, 0)  # UTC, about an hour after the fixture's TLE epoch
HOURS = 48
STEP_SECONDS = 2  # fine enough to catch even a ~1-minute pass; coarse enough to stay fast


@pytest.fixture(scope="module")
def samples(iss):
    """The ISS's altitude above West Lafayette every STEP_SECONDS. The mask doesn't affect
    altitude, so we compute this once and reuse it for every mask we test."""
    seconds = np.arange(0, HOURS * 3600 + 1, STEP_SECONDS)
    times = ts.utc(*START, seconds)
    altitude = (iss - WEST_LAFAYETTE.position()).at(times).altaz()[0].degrees
    return times, altitude


def brute_force_passes(times, altitude, mask_deg: float):
    """Return (rise, peak, set, max_elevation) for each complete run of samples above the mask."""
    above = altitude >= mask_deg

    passes = []
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        end = i
        while end + 1 < len(above) and above[end + 1]:
            end += 1
        # Like find_passes(), skip runs cut off by the start or end of the window.
        if i > 0 and end < len(above) - 1:
            peak = i + int(np.argmax(altitude[i:end + 1]))
            passes.append((times[i].utc_datetime(), times[peak].utc_datetime(),
                           times[end].utc_datetime(), altitude[peak]))
        i = end + 1
    return passes


@pytest.mark.parametrize("mask_deg", [0.0, 10.0, 30.0])
def test_find_passes_agrees_with_brute_force_sampling(iss, samples, mask_deg):
    station = replace(WEST_LAFAYETTE, min_elevation_deg=mask_deg)
    predicted = find_passes(iss, station, ts.utc(*START), HOURS)
    sampled = brute_force_passes(*samples, mask_deg)

    # Same number of passes: find_events() didn't miss (or invent) any.
    assert len(predicted) == len(sampled) > 0

    # A sampled rise is the first sample *after* the true crossing, so it can be up to one
    # step late; find_events() is good to half a second. Hence the step + 1 s tolerance.
    tolerance = timedelta(seconds=STEP_SECONDS + 1)
    for p, (rise, peak, set_, max_elevation) in zip(predicted, sampled):
        assert abs(p.rise - rise) <= tolerance
        assert abs(p.set - set_) <= tolerance
        assert abs(p.culmination - peak) <= tolerance
        # At the peak the altitude is barely changing, so sampling gets very close to the max.
        assert p.max_elevation_deg == pytest.approx(max_elevation, abs=0.01)
        assert p.max_elevation_deg >= max_elevation - 1e-9  # the true peak is never lower
