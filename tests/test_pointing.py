"""Pass profiles: pointing angles, range rate and Doppler, checked against physics (offline)."""

import csv
import io
from datetime import timedelta

import numpy as np
import pytest
from skyfield.api import load

from orbittrack.passes import find_passes
from orbittrack.pointing import (
    ISS_DOWNLINK_HZ, SPEED_OF_LIGHT_KM_S, doppler_shift_hz, pass_profile, pointing_csv,
)
from orbittrack.station import WEST_LAFAYETTE

ts = load.timescale()
START = ts.utc(2026, 10, 18, 12)


@pytest.fixture(scope="module")
def passes(iss):
    return find_passes(iss, WEST_LAFAYETTE, START, 24)


@pytest.fixture(scope="module")
def profiles(iss, passes):
    return [pass_profile(iss, WEST_LAFAYETTE, p) for p in passes]


def test_profile_runs_rise_to_set_every_second(passes, profiles):
    for p, prof in zip(passes, profiles):
        assert prof.times[0] == p.rise and prof.times[-1] == p.set
        assert prof.times[1] - prof.times[0] == timedelta(seconds=1)
        assert prof.elevation_deg[0] == pytest.approx(WEST_LAFAYETTE.min_elevation_deg, abs=0.05)
        assert prof.elevation_deg.max() == pytest.approx(p.max_elevation_deg, abs=0.05)


def test_range_rate_matches_the_change_in_range(profiles):
    # Range rate comes from the relative velocity; it must agree with differencing the range.
    for prof in profiles:
        numeric = np.diff(prof.range_km) / 1.0  # samples are 1 s apart (except possibly the last)
        midpoint = (prof.range_rate_km_s[:-2] + prof.range_rate_km_s[1:-1]) / 2
        assert numeric[:-1] == pytest.approx(midpoint, abs=0.005)


def test_doppler_is_positive_while_approaching_and_negative_while_receding(passes, profiles):
    for p, prof in zip(passes, profiles):
        assert prof.doppler_hz[0] > 0 > prof.doppler_hz[-1]
        # Zero shift happens at closest approach, which is (within a second or two) the peak.
        zero = prof.times[int(np.argmin(np.abs(prof.range_rate_km_s)))]
        assert abs((zero - p.culmination).total_seconds()) < 2


def test_doppler_never_exceeds_the_orbital_speed_limit(profiles):
    # The line-of-sight speed can't exceed the ISS's ~7.7 km/s orbital speed (plus Earth's spin).
    bound = ISS_DOWNLINK_HZ * 8.2 / SPEED_OF_LIGHT_KM_S
    for prof in profiles:
        assert np.abs(prof.doppler_hz).max() < bound


def test_overhead_pass_has_the_largest_doppler_swing_and_the_keyhole(passes, profiles):
    overhead = max(range(len(passes)), key=lambda i: passes[i].max_elevation_deg)
    assert passes[overhead].max_elevation_deg > 80
    # About +/-3.3 kHz at 145.8 MHz for the 84.9 deg pass.
    assert profiles[overhead].doppler_hz[0] == pytest.approx(3300, abs=100)
    # Near the zenith the antenna has to swing very fast in azimuth (measured ~11 deg/s),
    # far more than any lower pass needs.
    assert profiles[overhead].azimuth_rate_deg_s.max() > 5
    for i, prof in enumerate(profiles):
        if i != overhead:
            assert prof.azimuth_rate_deg_s.max() < 2


def test_doppler_formula():
    assert doppler_shift_hz(100e6, np.array([-3.0, 0.0, 3.0])) == pytest.approx(
        [100e6 * 3 / SPEED_OF_LIGHT_KM_S, 0, -100e6 * 3 / SPEED_OF_LIGHT_KM_S])


def test_frequency_scales_the_shift(iss, passes):
    low = pass_profile(iss, WEST_LAFAYETTE, passes[0], frequency_hz=145.8e6)
    high = pass_profile(iss, WEST_LAFAYETTE, passes[0], frequency_hz=437.8e6)
    assert high.doppler_hz == pytest.approx(low.doppler_hz * 437.8 / 145.8)


def test_pointing_csv_has_one_row_per_sample(profiles):
    prof = profiles[0]
    rows = list(csv.reader(io.StringIO(pointing_csv(prof))))
    assert rows[0] == ["utc", "azimuth_deg", "elevation_deg", "range_km", "range_rate_km_s",
                       "doppler_hz", "receive_frequency_hz"]
    assert len(rows) == len(prof.times) + 1
    first = rows[1]
    assert first[0].endswith("Z") and float(first[2]) == pytest.approx(10.0, abs=0.05)
    assert int(first[6]) == round(ISS_DOWNLINK_HZ + prof.doppler_hz[0])
