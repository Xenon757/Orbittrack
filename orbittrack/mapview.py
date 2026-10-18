"""Draw a satellite's ground track, the station's coverage, and predicted passes on an interactive map."""

import html
import statistics

import folium
from skyfield.api import EarthSatellite
from skyfield.timelib import Time

from orbittrack.groundtrack import circle_around, coverage_radius_deg, ground_track, split_at_antimeridian
from orbittrack.passes import Pass
from orbittrack.station import GroundStation

TRACK_COLOR = "#3b6fb6"
PASS_COLOR = "#e8590c"
STATION_COLOR = "#2b8a3e"


def build_map(
    satellite: EarthSatellite, station: GroundStation, start: Time, end: Time, passes: list[Pass]
) -> folium.Map:
    """Return a folium map; the caller decides where to save it.

    Layers: the full ground track (thin blue), the station and its coverage circle (green),
    each predicted pass highlighted (thick orange), and the satellite's position at `start`.
    """
    zone = station.local_zone()
    sat_name = html.escape(satellite.name.strip())
    fmap = folium.Map(location=[station.latitude_deg, station.longitude_deg], zoom_start=3, tiles="OpenStreetMap")

    track = ground_track(satellite, start, end)
    for segment in split_at_antimeridian(track):
        folium.PolyLine([(p.latitude_deg, p.longitude_deg) for p in segment],
                        color=TRACK_COLOR, weight=1.5, opacity=0.7).add_to(fmap)

    # The satellite's height changes slightly around its orbit; the average is fine for drawing.
    mean_height_km = statistics.fmean(p.height_km for p in track)
    radius_deg = coverage_radius_deg(mean_height_km, station.min_elevation_deg)
    folium.Polygon(
        circle_around(station.latitude_deg, station.longitude_deg, radius_deg),
        color=STATION_COLOR, weight=1.5, fill=True, fill_opacity=0.08,
        tooltip=f"Coverage above {station.min_elevation_deg:g}° (~{radius_deg * 111.2:.0f} km radius)",
    ).add_to(fmap)
    folium.CircleMarker(
        [station.latitude_deg, station.longitude_deg], radius=6, color=STATION_COLOR, fill=True, fill_opacity=1,
        tooltip=html.escape(station.name),
    ).add_to(fmap)

    for number, p in enumerate(passes, start=1):
        rise = p.rise.astimezone(zone)
        label = (f"Pass {number}: {rise:%Y-%m-%d %H:%M} {rise:%Z}, "
                 f"max {p.max_elevation_deg:.1f}°, {p.duration.total_seconds() / 60:.1f} min")
        pass_track = ground_track(satellite, start.ts.from_datetime(p.rise), start.ts.from_datetime(p.set), step_seconds=5)
        for segment in split_at_antimeridian(pass_track):
            folium.PolyLine([(q.latitude_deg, q.longitude_deg) for q in segment],
                            color=PASS_COLOR, weight=5, tooltip=label).add_to(fmap)

    now = track[0]
    folium.Marker(
        [now.latitude_deg, now.longitude_deg],
        tooltip=f"{sat_name} at {now.time.astimezone(zone):%Y-%m-%d %H:%M %Z} ({now.height_km:.0f} km up)",
        icon=folium.Icon(color="blue", icon="satellite", prefix="fa"),
    ).add_to(fmap)

    title = (f"<b>{sat_name}</b> ground track, {track[0].time.astimezone(zone):%Y-%m-%d %H:%M} to "
             f"{track[-1].time.astimezone(zone):%Y-%m-%d %H:%M %Z} &middot; "
             f"<span style='color:{PASS_COLOR}'>&#9644;</span> passes over {html.escape(station.name)}")
    fmap.get_root().html.add_child(folium.Element(
        f"<div style='position:fixed;top:10px;left:50px;z-index:1000;background:white;padding:6px 10px;"
        f"border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,.3);font:13px sans-serif'>{title}</div>"
    ))
    return fmap
