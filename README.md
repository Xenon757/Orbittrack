# OrbitTrack

A command-line satellite pass predictor for a ground station. It downloads current orbital
elements (TLEs) from [CelesTrak](https://celestrak.org), propagates them with SGP4, and
predicts when satellites rise above, peak over, and set below a station's elevation mask.
It can also schedule contacts across several satellites and stations, draw a satellite's
ground track on an interactive map, and serve predictions through a small web app and JSON API.

## Features

- **Current position**: altitude (elevation), azimuth and range from the station, plus the
  sub-satellite point and height above Earth.
- **Pass prediction**: every complete pass above the elevation mask in a time window, with
  rise / peak / set times, max elevation, rise→set compass direction and duration.
- **Any satellite, any station**: choose a satellite by name or NORAD ID from any CelesTrak
  group, and a station by latitude / longitude / elevation / time zone (default: West
  Lafayette, IN, with a 10° mask).
- **Contact scheduling**: for several satellites and stations, lists every contact in time
  order, drops contacts that overlap another at the same station (one antenna can track one
  satellite at a time), and reports the next scheduled contact.
- **Ground-track map**: an HTML map (folium / Leaflet) showing the ground track, the station's
  coverage circle, each predicted pass, and the satellite's position at the start time.
- **Web app**: a FastAPI site with a prediction form, the pass table, the embedded map, and the
  same data as JSON at `/api/passes` (interactive API docs at `/docs`).
- **Polite data handling**: TLE downloads are cached locally for 6 hours; if a download fails,
  a stale cached copy is used with a warning.

## Setup

Tested on Python 3.12.

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt   # app + test tools; the server only needs requirements.txt
```

## Usage

```bash
python -m orbittrack.cli                                   # ISS from West Lafayette, next 24 h
python -m orbittrack.cli --sat "CSS (TIANHE)" --hours 48 --min-elevation 20
python -m orbittrack.cli --sat 25544 --lat 51.48 --lon 0 --tz Europe/London
python -m orbittrack.cli --group weather --sat "NOAA 20 (JPSS-1)"
python -m orbittrack.cli --hours 6 --map iss_track.html    # also save a ground-track map
python -m orbittrack.cli --sat 25544 --sat "CSS (TIANHE)" \
    --station west-lafayette --station "Houston,29.5593,-95.09,10,America/Chicago"
```

| Option | Meaning |
|---|---|
| `--sat` | Satellite name or NORAD ID; repeat for several (default: 25544, the ISS) |
| `--group` | CelesTrak group to download, e.g. `stations`, `weather` (default: `stations`) |
| `--tle-file` | Read TLEs from a local file instead of downloading |
| `--lat`, `--lon`, `--elev` | A single custom station location |
| `--station` | `west-lafayette` or `NAME,LAT,LON[,ELEV_M[,TIME_ZONE]]`; repeat for several |
| `--tz` | IANA time zone for displayed times (default: the station's; UTC for several stations) |
| `--start` | Start of the search window, ISO 8601; no offset means UTC (default: now) |
| `--hours` | Length of the search window (default: 24) |
| `--min-elevation` | Elevation mask in degrees (default: 10) |
| `--map FILE` | Save a ground-track map (one satellite and one station only) |

Invalid input (out-of-range coordinates, unknown time zones, malformed times or stations)
is rejected with a usage error. An unknown satellite name suggests close matches, e.g.
`--sat "NOAA 20"` suggests `NOAA 20 (JPSS-1)`.

### Web app

```bash
uvicorn orbittrack.web:app --reload
```

Then open <http://localhost:8000>. The page has a form for the satellite, CelesTrak group,
station location, time window and mask (plus a "Use my location" button that fills in the
browser's coordinates and time zone), and shows the current position, next pass, pass table and
ground-track map. Every parameter is in the URL, so results can be bookmarked or shared.

The same parameters work on the JSON endpoint:

```bash
curl "http://localhost:8000/api/passes?sat=25544&lat=40.4237&lon=-86.9212&hours=24"
```

It returns the satellite and TLE age, the station, the current look angles and sub-satellite
point, and each pass with UTC ISO 8601 rise / culmination / set times, max elevation, rise and
set azimuths, and duration. Because the app is meant to face the public:

- Only six CelesTrak groups are accepted (`stations`, `visual`, `weather`, `amateur`, `science`,
  `gps-ops`). The group name becomes part of a cache file path and a CelesTrak request, so free
  text could be used for path tricks or to make the server spam CelesTrak.
- Parameters are range-checked (a bad value gets HTTP 422), and the window is capped at 72 hours.
- All visitors share one TLE cache. A lock ensures only one request re-downloads a stale file,
  and each file is parsed once and reused until it changes.
- An unknown satellite returns 404 with name suggestions; a failed download with nothing
  cached returns 503.

### Example output

These runs use the TLE file checked into `tests/fixtures/` and a fixed start time, so they are
reproducible:

```
$ python -m orbittrack.cli --tle-file tests/fixtures/stations_2026-10-18.tle --start 2026-10-18T12:00
Satellite:  ISS (ZARYA) (NORAD 25544)
TLE epoch:  2026-10-18 11:12:47 UTC (0.03 days from start)
Station:    West Lafayette, IN (40.4237, -86.9212, 187 m)
Time:       2026-10-18 12:00:00 UTC

Altitude:    -52.57°  -> below the horizon
Azimuth:      77.20°
Range:        10653 km

Sub-satellite point: -2.37, 24.74 at 421 km above Earth

Passes above 10° in the next 24 h (times in America/Indiana/Indianapolis):

   #  Date            Rise      Peak       Set  Max el  Direction  Duration
   1  2026-10-18  09:06:50  09:08:25  09:09:59   12.6°  NW -> NE       3m09s
   2  2026-10-18  10:43:45  10:46:16  10:48:46   18.8°  NW -> NE       5m01s
   3  2026-10-18  12:20:05  12:23:30  12:26:53   84.9°  NW -> SE       6m48s
   4  2026-10-18  13:59:05  13:59:49  14:00:34   10.5°  SW -> SW       1m30s
   5  2026-10-19  05:03:10  05:06:16  05:09:24   35.0°  SW -> E        6m14s
   6  2026-10-19  06:40:03  06:43:07  06:46:11   30.6°  W -> NE        6m08s
```

Contact scheduling for three satellites at two stations. Tianzhou-10 is a cargo ship docked to
the Chinese station (CSS), so their passes coincide and each Tianzhou contact is dropped:

```
$ python -m orbittrack.cli --tle-file tests/fixtures/stations_2026-10-18.tle --start 2026-10-18T12:00 \
    --sat 25544 --sat "CSS (TIANHE)" --sat TIANZHOU-10 \
    --station west-lafayette --station "Houston,29.5593,-95.09,10,America/Chicago" --hours 16
Contacts above 10° in the next 16 h for 3 satellite(s) at 2 station(s) (times in UTC):

   #  Date            Rise       Set  Max el  Duration  Satellite     Station             Status
   1  2026-10-18  13:06:50  13:09:59   12.6°      3m09s  ISS (ZARYA)   West Lafayette, IN  scheduled
   2  2026-10-18  14:43:45  14:48:46   18.8°      5m01s  ISS (ZARYA)   West Lafayette, IN  scheduled
   3  2026-10-18  16:20:05  16:26:53   84.9°      6m48s  ISS (ZARYA)   West Lafayette, IN  scheduled
   4  2026-10-18  17:57:29  18:04:15   87.5°      6m46s  ISS (ZARYA)   Houston             scheduled
   5  2026-10-18  17:59:05  18:00:34   10.5°      1m29s  ISS (ZARYA)   West Lafayette, IN  scheduled
   6  2026-10-19  02:08:12  02:14:27   62.4°      6m16s  CSS (TIANHE)  Houston             scheduled
   7  2026-10-19  02:08:12  02:14:27   62.4°      6m16s  TIANZHOU-10   Houston             conflict: station busy with #6
   8  2026-10-19  02:11:55  02:16:52   21.0°      4m57s  CSS (TIANHE)  West Lafayette, IN  scheduled
   9  2026-10-19  02:11:55  02:16:52   21.0°      4m57s  TIANZHOU-10   West Lafayette, IN  conflict: station busy with #8
  10  2026-10-19  03:45:56  03:50:46   19.6°      4m50s  CSS (TIANHE)  Houston             scheduled
  11  2026-10-19  03:45:56  03:50:46   19.6°      4m50s  TIANZHOU-10   Houston             conflict: station busy with #10
  12  2026-10-19  03:47:47  03:54:07   79.1°      6m20s  CSS (TIANHE)  West Lafayette, IN  scheduled
  13  2026-10-19  03:47:47  03:54:07   79.1°      6m20s  TIANZHOU-10   West Lafayette, IN  conflict: station busy with #12

Scheduled 9 of 13 contacts; 4 dropped because the station was already busy.
Next contact: #1 ISS (ZARYA) over West Lafayette, IN at 2026-10-18 13:06:50 UTC (in 1h 07m), max 12.6°
```

## How it works

| Module | Responsibility |
|---|---|
| `tle.py` | Download a CelesTrak group, cache it for 6 h, parse it, and look up satellites by NORAD ID or exact name |
| `station.py` | Ground station: WGS84 position, elevation mask, display time zone |
| `observe.py` | Topocentric look angles (altitude / azimuth / range) and the sub-satellite point |
| `passes.py` | Pass prediction with Skyfield's `find_events()`, grouped into complete rise → peak → set passes |
| `schedule.py` | Contacts across satellites × stations, and per-station conflict resolution |
| `groundtrack.py` | Ground-track sampling, antimeridian splitting, and the station's coverage circle |
| `mapview.py` | Folium map rendering |
| `cli.py` | Argument parsing, validation, and table output |
| `web.py`, `templates/` | FastAPI routes (`/`, `/map`, `/api/passes`), parameter validation, shared TLE cache |

Key details:

- **Propagation**: SGP4 (via Skyfield) on TLE elements. Predictions get less accurate the
  further you are from the TLE epoch, so the CLI prints the TLE's age.
- **Pass finding**: `find_events()` locates each mask crossing to within half a second
  rather than stepping through time. Passes already in progress at the start of the window,
  or still in progress at the end, are left out because their true rise or set is unknown.
- **Conflict scheduling**: earliest-finish-first greedy per station, which maximizes the
  *number* of non-overlapping contacts at each station. It does not consider pass quality or
  priority, so it can keep a low pass over a better one that ends later.
- **Coverage circle**: drawn using the central-angle formula
  `arccos(R·cos(mask) / (R + h)) − mask` on a spherical Earth, at the satellite's average
  height over the window.
- **Times**: computed in UTC; converted to the station's time zone only for display.

## Testing

```bash
pytest
```

107 tests, all offline. They use a real CelesTrak TLE file from 2026-10-18 checked into
`tests/fixtures/` and fixed start times, so results are deterministic. CLI tests fail if
anything tries to download, and web tests swap in the fixture through FastAPI's dependency
overrides. The tests include:

- **Independent cross-check**: pass predictions are compared against brute-force altitude
  sampling every 2 s over 48 h at 0°, 10° and 30° masks: same pass count, times within 3 s,
  max elevation within 0.01°.
- **Geometry checks**: a station directly under the satellite sees it at 90° with range equal
  to its height; a station on the opposite side of the Earth sees it at −90°; a station on the
  coverage circle sees it at the mask (within 0.2°).
- **Physical limits**: the ISS (51.6° inclination) never rises at 80°N and is never above 10°
  from Fairbanks (64.8°N); its ground-track latitude stays within the inclination; ground speed
  is about 6.9 km/s.
- **Pass properties**: rise < peak < set, max elevation ≥ mask, sane durations, and a window
  starting mid-pass drops only that pass.
- **Scheduling**: conflict rules on synthetic contacts, and a comparison against brute force
  over all subsets confirming the greedy schedule is as large as possible.
- **CLI and map**: options, input validation, schedule output, and generated map content.
- **Web app**: the API returns the same passes as the core code; the page, map and error pages
  render; bad parameters (including disallowed groups) get 422; visitor input is HTML-escaped;
  an outage gives 503; and the shared cache parses a file once until it changes.

## Limitations

- One CelesTrak group (or one TLE file) per run, so satellites from different groups can't be
  mixed in one schedule.
- The map supports one satellite and one station.
- The web app handles one satellite and one station per request; multi-satellite contact
  scheduling is CLI-only. It is set up to run locally and is not deployed anywhere.
- Contact scheduling ignores pass quality, priorities and antenna slew/setup time between contacts.
- Atmospheric refraction is not modeled.
- Accuracy is limited by SGP4 and the age of the TLE; always use fresh elements.

## Data

Orbital elements come from [CelesTrak](https://celestrak.org/NORAD/elements/). Cached
downloads are stored in `cache/` and are not committed.
