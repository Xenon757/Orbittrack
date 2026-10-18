"""TLE loading, lookup and caching. No test here touches the network."""

import os
import time
import urllib.request

import pytest
from skyfield.api import load

from orbittrack.tle import fetch_tle_file, find_satellite, load_satellites, tle_age_days

@pytest.fixture
def satellites(tle_path):
    return load_satellites(tle_path)


def no_network(*args, **kwargs):
    raise OSError("network disabled in tests")


def test_loads_every_record_in_the_file(satellites):
    assert len(satellites) == 17


def test_find_by_norad_id(satellites):
    assert find_satellite(satellites, "25544").name.strip() == "ISS (ZARYA)"


def test_find_by_name_is_case_insensitive(satellites):
    assert find_satellite(satellites, "css (tianhe)").model.satnum == 48274


def test_unknown_satellite_raises(satellites):
    with pytest.raises(LookupError):
        find_satellite(satellites, "NOT A SATELLITE")


def test_tle_age_is_zero_at_epoch(satellites):
    iss = find_satellite(satellites, "25544")
    assert tle_age_days(iss, iss.epoch) == pytest.approx(0.0)
    assert tle_age_days(iss, load.timescale().utc(2026, 10, 20, 11, 12, 47)) == pytest.approx(2.0, abs=1e-3)


def test_fresh_cache_is_used_without_downloading(tle_path, tmp_path, monkeypatch):
    cached = tmp_path / "stations.tle"
    cached.write_bytes(tle_path.read_bytes())
    monkeypatch.setattr(urllib.request, "urlopen", no_network)

    assert fetch_tle_file("stations", cache_dir=tmp_path) == cached


def test_stale_cache_is_used_if_download_fails(tle_path, tmp_path, monkeypatch):
    cached = tmp_path / "stations.tle"
    cached.write_bytes(tle_path.read_bytes())
    ten_hours_ago = time.time() - 10 * 3600
    os.utime(cached, (ten_hours_ago, ten_hours_ago))
    monkeypatch.setattr(urllib.request, "urlopen", no_network)

    assert fetch_tle_file("stations", cache_dir=tmp_path, max_age_hours=6) == cached


def test_no_cache_and_no_network_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    with pytest.raises(OSError):
        fetch_tle_file("stations", cache_dir=tmp_path)
