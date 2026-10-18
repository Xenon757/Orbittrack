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
    ["--station", "x,1"],  # too few fields
    ["--station", "Bad,95,0"],  # impossible latitude
    ["--station", "x,a,b"],  # not numbers
    ["--station", "x,1,1,0,Mars/Olympus"],
    ["--station", "west-lafayette", "--lat", "1", "--lon", "1"],  # two ways to give a location
    ["--sat", "25544", "--sat", "48274", "--map", "x.html"],  # map is single-satellite only
])
def test_bad_input_is_rejected_with_usage_error(capsys, bad_args):
    with pytest.raises(SystemExit) as exit_info:
        run(capsys, *bad_args)
    assert exit_info.value.code == 2


def test_start_without_offset_is_utc_and_offsets_are_kept():
    assert cli.utc_datetime("2026-10-18T12:00").tzinfo == timezone.utc
    assert cli.utc_datetime("2026-10-18T08:00-04:00").utcoffset() == timedelta(hours=-4)


# --- several satellites / stations: the combined contact schedule ---

HOUSTON = "Houston,29.5593,-95.09,10,America/Chicago"
THREE_SATS = ["--sat", "25544", "--sat", "CSS (TIANHE)", "--sat", "TIANZHOU-10"]


def test_schedule_lists_every_contact_and_drops_conflicts(capsys):
    out = run(capsys, *THREE_SATS, "--station", "west-lafayette", "--station", HOUSTON)
    assert "for 3 satellite(s) at 2 station(s) (times in UTC)" in out
    assert len(PASS_ROW.findall(out)) == 29
    # Tianzhou-10 is docked to CSS, so every one of its 10 contacts collides with a CSS contact.
    assert "Scheduled 19 of 29 contacts; 10 dropped because the station was already busy." in out
    assert out.count("conflict: station busy with #") == 10
    assert "Next contact: #1 ISS (ZARYA) over West Lafayette, IN at 2026-10-18 13:06:50 UTC (in 1h 07m)" in out


def test_schedule_at_one_station_uses_its_time_zone(capsys):
    out = run(capsys, "--sat", "25544", "--sat", "48274")
    assert "at 1 station(s) (times in America/Indiana/Indianapolis)" in out


def test_tz_overrides_the_schedule_time_zone(capsys):
    out = run(capsys, "--sat", "25544", "--station", "west-lafayette", "--station", HOUSTON, "--tz", "Asia/Tokyo")
    assert "(times in Asia/Tokyo)" in out


def test_same_satellite_twice_is_counted_once(capsys):
    out = run(capsys, "--sat", "25544", "--sat", "ISS (ZARYA)")
    assert "Satellite:  ISS (ZARYA) (NORAD 25544)" in out  # back to the single-satellite view


def test_station_spec_parses_presets_and_custom_stations():
    assert cli.station_spec("West-Lafayette") == cli.WEST_LAFAYETTE
    minimal = cli.station_spec("Houston, 29.56, -95.09")
    assert (minimal.name, minimal.latitude_deg, minimal.elevation_m, minimal.timezone) == ("Houston", 29.56, 0.0, "UTC")
    full = cli.station_spec(HOUSTON)
    assert (full.elevation_m, full.timezone) == (10.0, "America/Chicago")
