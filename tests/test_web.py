"""Web app tests: JSON API, HTML page, map route, input validation and the shared TLE cache.

The app's TLE loader is swapped for the fixture file via FastAPI's dependency overrides,
so these tests never download anything.
"""

import os
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from orbittrack import web
from orbittrack.passes import find_passes
from orbittrack.station import WEST_LAFAYETTE
from orbittrack.tle import load_satellites

FIXED = {"start": "2026-10-18T12:00"}  # about an hour after the fixture's TLE epoch


@pytest.fixture
def client(tle_path):
    satellites = load_satellites(tle_path)
    web.app.dependency_overrides[web.get_group_loader] = lambda: (lambda group: satellites)
    yield TestClient(web.app)
    web.app.dependency_overrides.clear()


@pytest.fixture
def offline_client():
    """A client whose TLE loader fails the way a CelesTrak outage with no cache would."""
    def unreachable(group):
        raise OSError("network down")
    web.app.dependency_overrides[web.get_group_loader] = lambda: unreachable
    yield TestClient(web.app)
    web.app.dependency_overrides.clear()


# --- JSON API ---

def test_api_returns_the_same_passes_as_the_core_code(client, iss):
    response = client.get("/api/passes", params=FIXED)
    assert response.status_code == 200
    data = response.json()

    assert data["satellite"]["name"] == "ISS (ZARYA)" and data["satellite"]["norad_id"] == 25544
    assert data["station"]["name"] == "West Lafayette, IN"
    assert data["window"] == {"start": "2026-10-18T12:00:00Z", "end": "2026-10-19T12:00:00Z"}

    expected = find_passes(iss, WEST_LAFAYETTE, web.ts.utc(2026, 10, 18, 12), 24)
    assert len(data["passes"]) == len(expected) == 6
    assert data["passes"][2]["max_elevation_deg"] == round(expected[2].max_elevation_deg, 2)
    assert data["passes"][2]["rise"] == expected[2].rise.isoformat().replace("+00:00", "Z")


def test_api_accepts_a_custom_station_and_mask(client):
    data = client.get("/api/passes", params={**FIXED, "lat": 29.5593, "lon": -95.09, "station": "Houston",
                                             "tz": "America/Chicago", "min_elevation": 30}).json()
    assert data["station"]["name"] == "Houston"
    assert all(p["max_elevation_deg"] >= 30 for p in data["passes"])


def test_api_unknown_satellite_is_404_with_suggestions(client):
    response = client.get("/api/passes", params={"sat": "iss"})
    assert response.status_code == 404
    assert "Did you mean: ISS (ZARYA), ISS (NAUKA)?" in response.json()["detail"]


@pytest.mark.parametrize("bad", [
    {"group": "../../etc"},  # not an allowed CelesTrak group: never reaches the file system
    {"group": "active"},  # a real group, but not on the allowlist
    {"lat": 95},
    {"lon": -181},
    {"tz": "Mars/Olympus"},
    {"hours": 0},
    {"hours": web.MAX_HOURS + 1},  # caps the work one request can ask for
    {"min_elevation": 90},
    {"start": "yesterday"},
    {"sat": "x" * 61},
])
def test_api_rejects_bad_parameters(client, bad):
    assert client.get("/api/passes", params=bad).status_code == 422


def test_api_reports_503_when_celestrak_is_unreachable(offline_client):
    assert offline_client.get("/api/passes").status_code == 503


# --- HTML page ---

def test_page_shows_passes_next_pass_and_map(client):
    response = client.get("/", params=FIXED)
    assert response.status_code == 200
    page = response.text
    assert "ISS (ZARYA)" in page and "NORAD 25544" in page
    assert page.count("<tr>") == 7  # header row + 6 passes
    assert "2026-10-18 09:06 EDT" in page  # next pass rise, in the station's time zone
    assert '<iframe src="/map?' in page and "start=2026-10-18T12%3A00%3A00%2B00%3A00" in page
    assert '<option value="CSS (TIANHE)">' in page  # satellite name suggestions for the group


def test_page_with_unknown_satellite_explains_the_problem(client):
    response = client.get("/", params={"sat": "iss"})
    assert response.status_code == 404
    assert "Did you mean: ISS (ZARYA), ISS (NAUKA)?" in response.text
    assert "<iframe" not in response.text


def test_page_escapes_user_input(client):
    # Anything a visitor types is shown back on the page, so it must be HTML-escaped.
    response = client.get("/", params={"sat": "<script>alert(1)</script>"})
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text


def test_page_explains_an_outage(offline_client):
    response = offline_client.get("/")
    assert response.status_code == 503
    assert "Couldn&#39;t download orbital data from CelesTrak" in response.text


def test_map_route_renders_the_ground_track(client):
    response = client.get("/map", params={**FIXED, "hours": 6})
    assert response.status_code == 200
    assert "ISS (ZARYA)" in response.text and "Coverage above 10°" in response.text


def test_health_check_needs_no_orbital_data(offline_client):
    # Even with CelesTrak unreachable the app itself is healthy, so this must still say ok.
    response = offline_client.get("/healthz")
    assert response.status_code == 200 and response.json() == {"status": "ok"}


# --- the shared TLE cache ---

def test_load_group_parses_once_until_the_file_changes(tle_path, tmp_path, monkeypatch):
    cached_file = tmp_path / "stations.tle"
    shutil.copy(tle_path, cached_file)
    monkeypatch.setattr(web, "fetch_tle_file", lambda group: cached_file)
    monkeypatch.setattr(web, "_parsed", {})
    parses = []
    monkeypatch.setattr(web, "load_satellites", lambda path: parses.append(path) or load_satellites(path))

    first = web.load_group("stations")
    assert web.load_group("stations") is first  # same parsed list reused by every request
    assert len(parses) == 1

    later = time.time() + 60  # simulate a fresh download replacing the cached file
    os.utime(cached_file, (later, later))
    web.load_group("stations")
    assert len(parses) == 2


def test_failed_download_is_not_retried_for_a_while(monkeypatch):
    attempts = []

    def failing_fetch(group):
        attempts.append(group)
        raise OSError("timed out")

    clock = [1000.0]
    monkeypatch.setattr(web, "fetch_tle_file", failing_fetch)
    monkeypatch.setattr(web, "_parsed", {})
    monkeypatch.setattr(web, "_last_failure", {})
    monkeypatch.setattr(web.time, "monotonic", lambda: clock[0])

    for _ in range(3):  # the first visitor waits on the network; the next two get an instant 503
        with pytest.raises(OSError, match="timed out"):
            web.load_group("stations")
    assert len(attempts) == 1

    clock[0] += web.RETRY_AFTER_SECONDS + 1  # after the back-off, try CelesTrak again
    with pytest.raises(OSError):
        web.load_group("stations")
    assert len(attempts) == 2


def test_outage_reason_is_shown(offline_client):
    assert "(Reason: network down)" in offline_client.get("/api/passes").json()["detail"]
