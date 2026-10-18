"""Shared test fixtures. pytest makes these available to every test file automatically."""

from pathlib import Path

import pytest

from orbittrack.tle import find_satellite, load_satellites

# A real CelesTrak "stations" download from 2026-10-18 (ISS TLE epoch 2026-10-18 11:12:47 UTC).
# Using a fixed historical file keeps every test deterministic and offline.
STATIONS_TLE = Path(__file__).parent / "fixtures" / "stations_2026-10-18.tle"


@pytest.fixture(scope="session")
def tle_path() -> Path:
    return STATIONS_TLE


@pytest.fixture(scope="session")
def iss(tle_path):
    return find_satellite(load_satellites(tle_path), "25544")
