"""Web app tests: JSON API, HTML page, map route, input validation and the shared TLE cache.

The app's TLE loader is swapped for the fixture file via FastAPI's dependency overrides,
so these tests never download anything.
"""

import os
import shutil
import time
from pathlib import Path

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

def test_page_shows_passes_next_pass_sky_plot_and_map(client):
    response = client.get("/", params=FIXED)
    assert response.status_code == 200
    page = response.text
    assert "ISS (ZARYA)" in page and "NORAD 25544" in page
    assert page.count('<tr data-pass="') == 6  # one table row per pass
    assert page.count('<g class="pass') == 6  # and one path per pass on the sky plot
    assert "T−</small>1h 07m" in page  # countdown to the next pass from the fixed start time
    assert "Rises <strong>09:06</strong> Sun 18 Oct, EDT" in page  # in the station's time zone
    assert 'src="/map?' in page and "start=2026-10-18T12%3A00%3A00%2B00%3A00" in page
    assert '<option value="CSS (TIANHE)">' in page  # satellite name suggestions for the group


def test_sky_plot_places_each_peak_by_its_elevation(tle_path):
    # Looking straight up: the center is the zenith (90 deg) and the horizon is the outer circle,
    # so a peak's distance from the center is R * (90 - max elevation) / 90.
    satellites = load_satellites(tle_path)
    prediction = web.predict(web.PassQuery(start="2026-10-18T12:00"), lambda group: satellites)
    sky = web.sky_plot(prediction)

    assert len(sky["paths"]) == len(prediction.passes) == 6
    for path, p in zip(sky["paths"], prediction.passes):
        x, y = path["peak"]
        distance = ((x - web.SKY_CENTER) ** 2 + (y - web.SKY_CENTER) ** 2) ** 0.5
        assert distance == pytest.approx(web.SKY_RADIUS * (90 - p.max_elevation_deg) / 90, abs=0.5)
        # Both ends of every path sit on the mask ring.
        for end_x, end_y in (path["start"], path["end"]):
            ring = ((end_x - web.SKY_CENTER) ** 2 + (end_y - web.SKY_CENTER) ** 2) ** 0.5
            assert ring == pytest.approx(sky["mask_r"], abs=0.5)


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
    assert "Couldn&#39;t download orbital data right now" in response.text


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
    monkeypatch.setattr(web, "fetch_tle_file", lambda group, url_template: cached_file)
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

    def failing_fetch(group, url_template):
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


def test_load_group_downloads_from_the_configured_source(tle_path, monkeypatch):
    # ORBITTRACK_TLE_URL is read into TLE_SOURCE at startup; a deployment uses it to read a mirror.
    requested = []
    monkeypatch.setattr(web, "TLE_SOURCE", "https://mirror.example/{group}.tle")
    monkeypatch.setattr(web, "fetch_tle_file", lambda group, url_template: requested.append(url_template) or tle_path)
    monkeypatch.setattr(web, "_parsed", {})
    monkeypatch.setattr(web, "_last_failure", {})

    web.load_group("stations")
    assert requested == ["https://mirror.example/{group}.tle"]


def test_bad_data_from_the_source_is_treated_as_an_outage(monkeypatch):
    def not_tle(group, url_template):
        raise ValueError("did not return TLE data")
    monkeypatch.setattr(web, "fetch_tle_file", not_tle)
    monkeypatch.setattr(web, "_parsed", {})
    monkeypatch.setattr(web, "_last_failure", {})
    with pytest.raises(OSError, match="did not return TLE data"):
        web.load_group("stations")


def test_mirror_workflow_mirrors_exactly_the_allowed_groups():
    # The deployed app reads TLEs from the GitHub Pages mirror, so a group missing from the
    # workflow would 503 in production even though every local test passed.
    workflow = (Path(__file__).parent.parent / ".github" / "workflows" / "tle-mirror.yml").read_text()
    line = next(line for line in workflow.splitlines() if line.strip().startswith("GROUPS:"))
    mirrored = line.split(":", 1)[1].strip().strip('"').split()
    assert sorted(mirrored) == sorted(web.GROUP_LABELS)


# --- live display samples ---

@pytest.fixture(scope="module")
def fixed_prediction(tle_path):
    satellites = load_satellites(tle_path)
    return web.predict(web.PassQuery(start="2026-10-18T12:00"), lambda group: satellites)


def to_alt_az_range(east, north, up):
    """The same conversion the browser does."""
    import math
    distance = math.sqrt(east ** 2 + north ** 2 + up ** 2)
    return math.degrees(math.asin(up / distance)), math.degrees(math.atan2(east, north)) % 360, distance


def test_live_samples_cover_the_window_every_30_seconds(fixed_prediction):
    live = web.live_samples(fixed_prediction)
    assert live["step_s"] == 30 and live["duration_s"] == 24 * 3600
    assert len(live["samples"]) == 24 * 3600 // 30 + 1
    assert live["start_ms"] == round(fixed_prediction.start.utc_datetime().timestamp() * 1000)
    assert live["passes"] == [[round(x.rise.timestamp() * 1000), round(x.set.timestamp() * 1000)]
                              for x in fixed_prediction.passes]


def test_live_samples_match_the_direct_calculation(fixed_prediction, iss):
    live = web.live_samples(fixed_prediction)
    for index in (0, 1000, 2880):
        moment = fixed_prediction.start + index * 30 / 86400
        expected = web.look_angles(iss, fixed_prediction.station, moment)
        lat, lon, height, east, north, up = live["samples"][index]
        altitude, azimuth, distance = to_alt_az_range(east, north, up)
        assert altitude == pytest.approx(expected.altitude_deg, abs=0.01)
        assert azimuth == pytest.approx(expected.azimuth_deg, abs=0.05)
        assert distance == pytest.approx(expected.range_km, abs=0.2)


def test_interpolating_between_samples_points_within_a_tenth_of_a_degree(fixed_prediction, iss):
    # The browser shows positions between samples by straight-line interpolation of the
    # east/north/up vector. Halfway between samples is the worst case; while the satellite
    # is above the horizon, that must stay below the 0.1 deg the page displays.
    import numpy as np
    live = web.live_samples(fixed_prediction)
    samples = np.array([s[3:] for s in live["samples"]])
    halfway = fixed_prediction.start + (np.arange(len(samples) - 1) * 30 + 15) / 86400
    alt, az, km = (iss - fixed_prediction.station.position()).at(halfway).altaz()
    a, z, k = alt.radians, az.radians, km.km
    truth = np.stack([k * np.cos(a) * np.sin(z), k * np.cos(a) * np.cos(z), k * np.sin(a)], axis=1)
    guess = (samples[:-1] + samples[1:]) / 2
    cos = np.sum(guess * truth, axis=1) / (np.linalg.norm(guess, axis=1) * np.linalg.norm(truth, axis=1))
    error_deg = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    above = truth[:, 2] > 0
    assert above.any() and error_deg[above].max() < 0.1


def test_page_embeds_live_samples_as_json(client):
    import json
    import re
    page = client.get("/", params=FIXED).text
    match = re.search(r'<script type="application/json" id="live">(.*?)</script>', page, re.S)
    assert match, "live data block missing"
    assert len(json.loads(match.group(1))["samples"]) == 2881


def test_responses_are_gzipped_when_the_browser_accepts_it(client):
    response = client.get("/", params=FIXED, headers={"Accept-Encoding": "gzip"})
    assert response.headers.get("content-encoding") == "gzip"


def test_page_covers_the_map_with_a_loading_screen_until_it_is_ready(client):
    page = client.get("/", params=FIXED).text
    assert 'id="map-loading"' in page and 'class="map-loading"' in page  # visible until told otherwise
    assert '"orbittrack-map-ready"' in page and "event.origin === location.origin" in page
