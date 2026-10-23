"""Accuracy study: OEM parsing, the radial/along/cross frame, and SGP4 vs NASA on real data.

Uses the first 24 hours of NASA's ISS ephemeris published 2026-10-21 (public domain,
NASA JSC/FOD/TOPO), saved in tests/fixtures/, together with the 2026-10-18 TLE fixture.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from orbittrack import accuracy
from orbittrack.accuracy import compare, parse_oem, radial_along_cross, summary_rows

OEM_FIXTURE = Path(__file__).parent / "fixtures" / "iss_oem_2026-10-21_first_day.txt"

HEADER = """CCSDS_OEM_VERS = 2.0
META_START
OBJECT_NAME = ISS
CENTER_NAME = Earth
REF_FRAME = {frame}
TIME_SYSTEM = UTC
META_STOP
"""


@pytest.fixture(scope="module")
def ephemeris():
    return parse_oem(OEM_FIXTURE.read_text())


# --- parsing ---

def test_parses_every_state_vector_with_units_and_times(ephemeris):
    assert len(ephemeris.times) == 361  # 24 h at 4-minute spacing, both ends included
    assert ephemeris.times[0] == datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)
    assert ephemeris.times[-1] == datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)
    assert ephemeris.positions_km[0] == pytest.approx([5473.296487002460, 4023.630981315710, 231.432849155033])
    assert ephemeris.velocities_km_s[0] == pytest.approx([-2.97846913460243, 3.70269085133805, 6.01037025862305])
    # Units really are km and km/s: ISS orbital radius ~6,790-6,810 km, speed ~7.66 km/s.
    radii = np.linalg.norm(ephemeris.positions_km, axis=1)
    speeds = np.linalg.norm(ephemeris.velocities_km_s, axis=1)
    assert 6780 < radii.min() and radii.max() < 6820
    assert 7.6 < speeds.min() and speeds.max() < 7.7


def test_skips_comments_and_covariance_blocks():
    text = HEADER.format(frame="EME2000") + """COMMENT a note
2026-10-21T12:00:00.000 1 2 3 4 5 6
COVARIANCE_START
EPOCH = 2026-10-21T12:00:00.000
2026-10-21T12:00:00.000 9 9 9 9 9 9
COVARIANCE_STOP
2026-10-21T12:04:00.000 7 8 9 10 11 12
"""
    parsed = parse_oem(text)
    assert len(parsed.times) == 2
    assert parsed.positions_km[1] == pytest.approx([7, 8, 9])


@pytest.mark.parametrize("text", [
    HEADER.format(frame="TEME") + "2026-10-21T12:00:00.000 1 2 3 4 5 6\n",  # wrong frame
    HEADER.format(frame="EME2000"),  # no data
])
def test_rejects_data_it_cannot_compare_correctly(text):
    with pytest.raises(ValueError):
        parse_oem(text)


# --- radial / along-track / cross-track frame ---

def test_frame_components_on_a_simple_orbit():
    # Circular orbit along +x, moving toward +y: radial = x, along-track = y, cross-track = z.
    position = np.array([[7000.0, 0.0, 0.0]] * 3)
    velocity = np.array([[0.0, 7.5, 0.0]] * 3)
    differences = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    along, cross, radial = radial_along_cross(position, velocity, differences)
    assert list(radial) == pytest.approx([1, 0, 0])
    assert list(along) == pytest.approx([0, 1, 0])
    assert list(cross) == pytest.approx([0, 0, 1])


def test_frame_preserves_the_size_of_the_difference(ephemeris):
    rng = np.random.default_rng(7)
    differences = rng.normal(size=ephemeris.positions_km.shape)
    along, cross, radial = radial_along_cross(ephemeris.positions_km, ephemeris.velocities_km_s, differences)
    assert np.sqrt(along ** 2 + cross ** 2 + radial ** 2) == pytest.approx(np.linalg.norm(differences, axis=1))


# --- SGP4 vs NASA on real data ---

def test_three_day_old_tle_is_within_a_kilometre_of_nasa(iss, ephemeris):
    result = compare(iss, ephemeris)
    assert result.tle_age_days[0] == pytest.approx(3.03, abs=0.01)
    # Measured 0.80 km for this TLE against NASA's tracking-based state at the start of the file.
    assert result.total_km[0] == pytest.approx(0.80, abs=0.05)
    assert result.total_km.max() < 5  # stays within a few km across the first day


def test_error_is_almost_entirely_along_track(iss, ephemeris):
    result = compare(iss, ephemeris)
    share = np.sum(result.along_km ** 2) / np.sum(result.total_km ** 2)
    # Over this first day the total error is only ~1-3 km, so the few-hundred-metre cross-track
    # and radial parts still matter (measured 93.5%); over NASA's full 15 days it's 99.9%.
    assert share > 0.9
    assert np.abs(result.radial_km).max() < 1  # height is predicted to well under a kilometre
    # Timing error is the along-track error divided by speed (~7.66 km/s).
    assert result.timing_error_s == pytest.approx(result.along_km / result.speed_km_s)
    assert abs(result.timing_error_s[0]) < 0.2


def test_summary_rows_stop_at_the_end_of_the_file(iss, ephemeris):
    rows = summary_rows(compare(iss, ephemeris))
    assert [round((r["time"] - rows[0]["time"]).total_seconds() / 86400) for r in rows] == [0, 1]


# --- command line ---

def test_cli_reports_and_writes_csv(tle_path, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(accuracy, "fetch_tle_file", lambda group: pytest.fail("must not download"))
    out_csv = tmp_path / "diff.csv"
    accuracy.main(["--oem", str(OEM_FIXTURE), "--tle-file", str(tle_path), "--csv", str(out_csv)])
    out = capsys.readouterr().out
    assert "Measured at the OEM start (tracking-based): 0.80 km" in out
    assert "3.03-day-old TLE" in out
    assert len(out_csv.read_text().splitlines()) == 362  # header + 361 samples


def test_fetch_oem_file_rejects_anything_that_is_not_an_ephemeris(tmp_path, monkeypatch):
    import urllib.request

    class Page:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self.body

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: Page(b"<html>Access Denied</html>"))
    with pytest.raises(ValueError):
        accuracy.fetch_oem_file("https://example.test/oem.txt", cache_dir=tmp_path)
    assert not (tmp_path / "iss_oem.txt").exists()

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: Page(OEM_FIXTURE.read_bytes()))
    path = accuracy.fetch_oem_file("https://example.test/oem.txt", cache_dir=tmp_path)
    assert parse_oem(path.read_text()).times[0].year == 2026
