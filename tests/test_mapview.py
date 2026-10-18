"""Map rendering: checks the generated HTML has the expected layers (no network needed to build it)."""

from datetime import timedelta

import pytest
from skyfield.api import load

from orbittrack.mapview import PASS_COLOR, TRAIL_COLOR, build_map
from orbittrack.passes import find_passes
from orbittrack.station import WEST_LAFAYETTE

ts = load.timescale()
START = ts.utc(2026, 10, 18, 12, 0, 0)
END = START + timedelta(hours=6)


@pytest.fixture(scope="module")
def passes(iss):
    return find_passes(iss, WEST_LAFAYETTE, START, 6)


@pytest.fixture(scope="module")
def page(iss, passes):
    return build_map(iss, WEST_LAFAYETTE, START, END, passes).get_root().render()


def test_map_names_the_satellite_station_and_window(page):
    assert "ISS (ZARYA)" in page
    assert "West Lafayette, IN" in page
    assert "2026-10-18 08:00 to 2026-10-18 14:00 EDT" in page  # 12:00-18:00 UTC in local time


def test_map_labels_every_pass(page, passes):
    assert len(passes) == 3
    for number in range(1, len(passes) + 1):
        assert f"Pass {number}:" in page
    assert "max 84.9°" in page  # the near-overhead pass


def test_map_draws_coverage_circle_for_the_mask(page):
    assert "Coverage above 10°" in page


def test_map_with_no_passes_still_renders(iss):
    page = build_map(iss, WEST_LAFAYETTE, START, START + timedelta(hours=1), []).get_root().render()
    assert "ISS (ZARYA)" in page
    assert "Pass 1:" not in page
    assert PASS_COLOR in page  # only in the title legend


def test_map_animates_the_marker_from_the_track_samples(page):
    # The script moves the satellite marker along the same 30-second samples as the drawn track.
    assert "setLatLng" in page and "predicted position now" in page
    assert '"times_ms": [' in page
    assert "trail.setLatLngs" in page and f'color: "{TRAIL_COLOR}"' in page  # blue trail behind the marker
    assert "flown so far" in page  # and it's in the legend
