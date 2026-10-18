"""Draw a satellite's ground track, the station's coverage, and predicted passes on an interactive map."""

import html
import statistics

import folium
from skyfield.api import EarthSatellite
from skyfield.timelib import Time

from orbittrack.groundtrack import circle_around, coverage_radius_deg, ground_track, split_at_antimeridian
from orbittrack.passes import Pass
from orbittrack.station import GroundStation

# Same palette as the web page: ink on paper, one signal red for the passes.
INK = "#141414"
PAPER = "#F2F0EB"
TRACK_COLOR = INK
PASS_COLOR = "#D9381E"
STATION_COLOR = INK

# Monochrome tiles (keeps OpenStreetMap's attribution, drops its colors so the data stands out),
# square controls, and the page's typefaces.
MAP_STYLE = f"""
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap" rel="stylesheet">
<style>
  .leaflet-container {{ background: {PAPER}; font-family: "Archivo", "Helvetica Neue", Arial, sans-serif; }}
  .leaflet-tile-pane {{ filter: grayscale(1) contrast(0.92) brightness(1.04); }}
  .leaflet-bar, .leaflet-bar a {{ border-radius: 0 !important; }}
  .leaflet-bar {{ border: 2px solid {INK} !important; box-shadow: none !important; }}
  .leaflet-bar a {{ color: {INK}; }}
  .leaflet-tooltip {{ border-radius: 0; border: 1px solid {INK}; box-shadow: none; font: 600 12px/1.4 "Archivo", Arial, sans-serif; }}
  .ot-title {{ position: fixed; top: 10px; left: 54px; right: 10px; z-index: 1000; max-width: max-content;
    background: {INK}; color: {PAPER}; padding: 6px 10px; font: 700 11px/1.3 "Archivo", Arial, sans-serif;
    letter-spacing: 0.1em; text-transform: uppercase; }}
  .ot-title b {{ color: {PASS_COLOR}; }}
</style>
"""


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
                        color=TRACK_COLOR, weight=1.25, opacity=0.55).add_to(fmap)

    # The satellite's height changes slightly around its orbit; the average is fine for drawing.
    mean_height_km = statistics.fmean(p.height_km for p in track)
    radius_deg = coverage_radius_deg(mean_height_km, station.min_elevation_deg)
    folium.Polygon(
        circle_around(station.latitude_deg, station.longitude_deg, radius_deg),
        color=STATION_COLOR, weight=1.5, dash_array="3 5", fill=True, fill_color=STATION_COLOR, fill_opacity=0.05,
        tooltip=f"Coverage above {station.min_elevation_deg:g}° (~{radius_deg * 111.2:.0f} km radius)",
    ).add_to(fmap)
    folium.CircleMarker(
        [station.latitude_deg, station.longitude_deg], radius=5, color=STATION_COLOR, weight=2,
        fill=True, fill_color=PASS_COLOR, fill_opacity=1,
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
    folium.CircleMarker(
        [now.latitude_deg, now.longitude_deg], radius=7, color=INK, weight=2.5, fill=True, fill_color=PAPER, fill_opacity=1,
        tooltip=f"{sat_name} at {now.time.astimezone(zone):%Y-%m-%d %H:%M %Z} ({now.height_km:.0f} km up)",
    ).add_to(fmap)

    title = (f"{sat_name} ground track, {track[0].time.astimezone(zone):%Y-%m-%d %H:%M} to "
             f"{track[-1].time.astimezone(zone):%Y-%m-%d %H:%M %Z} &middot; "
             f"<b>&#9644; passes over {html.escape(station.name)}</b>")
    fmap.get_root().header.add_child(folium.Element(MAP_STYLE))
    fmap.get_root().html.add_child(folium.Element(f"<div class='ot-title'>{title}</div>"))
    return fmap
