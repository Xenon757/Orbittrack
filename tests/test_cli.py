"""End-to-end CLI tests: run main() with arguments against the fixture TLE (never the network)."""

import re
from datetime import timedelta, timezone

import pytest

from conftest import STATIONS_TLE
from orbittrack import cli

BASE_ARGS = ["--tle-file", str(STATIONS_TLE), "--start", "2026-10-18T12:00"]
PASS_ROW = re.compile(r"^\s+\d+\s+\d{4}-\d{2}-\d{2}\s", re.MULTILINE)


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch):
    """Fail loudly if any CLI test tries to download TLEs."""
    def refuse(*args, **kwargs):
        raise AssertionError("tests must not download TLEs")
    monkeypatch.setattr(cli, "fetch_tle_file", refuse)


def run(capsys, *args: str) -> str:
    cli.main([*BASE_ARGS, *args])
    return capsys.readouterr().out


def test_default_run_prints_iss_passes_for_west_lafayette(capsys):
    out = run(capsys)
    assert "ISS (ZARYA) (NORAD 25544)" in out
    assert "West Lafayette, IN" in out
    assert "times in America/Indiana/Indianapolis" in out
    assert len(PASS_ROW.findall(out)) == 6


def test_higher_mask_shows_fewer_passes(capsys):
    out = run(capsys, "--min-elevation", "30")
    assert "Passes above 30°" in out
    assert len(PASS_ROW.findall(out)) == 3


def test_satellite_can_be_chosen_by_name(capsys):
    out = run(capsys, "--sat", "css (tianhe)")
    assert "CSS (TIANHE) (NORAD 48274)" in out


def test_custom_station_and_time_zone(capsys):
    out = run(capsys, "--lat", "51.48", "--lon", "0", "--elev", "46", "--tz", "Europe/London")
    assert "Custom location (51.4800, 0.0000, 46 m)" in out
    assert "times in Europe/London" in out


def test_shorter_window_finds_fewer_passes(capsys):
    # The first pass after 12:00 UTC rises at 13:06 UTC, so 2 hours catches exactly one.
    assert len(PASS_ROW.findall(run(capsys, "--hours", "2"))) == 1


def test_window_with_no_passes_says_none(capsys):
    assert run(capsys, "--hours", "1").rstrip().endswith("none")


def test_map_option_writes_an_html_file(capsys, tmp_path):
    map_file = tmp_path / "track.html"
    out = run(capsys, "--hours", "6", "--map", str(map_file))
    assert f"Map saved to {map_file.resolve()}" in out
    assert "ISS (ZARYA)" in map_file.read_text()


def test_unknown_satellite_suggests_close_names(capsys):
    with pytest.raises(SystemExit) as exit_info:
        run(capsys, "--sat", "iss")
    assert exit_info.value.code == 1
    assert "Did you mean: ISS (ZARYA), ISS (NAUKA)?" in capsys.readouterr().err


@pytest.mark.parametrize("bad_args", [
    ["--lat", "40"],  # latitude without longitude
    ["--lat", "95", "--lon", "0"],  # impossible latitude
    ["--elev", "300"],  # height without a location
    ["--hours", "0"],
    ["--min-elevation", "90"],
    ["--start", "yesterday"],
    ["--tz", "Mars/Olympus"],
])
def test_bad_input_is_rejected_with_usage_error(capsys, bad_args):
    with pytest.raises(SystemExit) as exit_info:
        run(capsys, *bad_args)
    assert exit_info.value.code == 2


def test_start_without_offset_is_utc_and_offsets_are_kept():
    assert cli.utc_datetime("2026-10-18T12:00").tzinfo == timezone.utc
    assert cli.utc_datetime("2026-10-18T08:00-04:00").utcoffset() == timedelta(hours=-4)
