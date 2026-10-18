"""Command-line entry point: where is a satellite now, and when will it pass over my station?

Examples:
    python -m orbittrack.cli
    python -m orbittrack.cli --sat "CSS (TIANHE)" --hours 48 --min-elevation 20
    python -m orbittrack.cli --sat 25544 --lat 51.48 --lon -0.0 --tz Europe/London
    python -m orbittrack.cli --hours 6 --map iss_track.html
"""

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from skyfield.api import EarthSatellite, load
from skyfield.timelib import Time

from orbittrack.mapview import build_map
from orbittrack.observe import look_angles, sub_point
from orbittrack.passes import Pass, compass_direction, find_passes
from orbittrack.station import WEST_LAFAYETTE, GroundStation
from orbittrack.tle import fetch_tle_file, find_satellite, load_satellites, tle_age_days


# --- argument types: reject bad input with a clear message instead of a traceback ---

def bounded_float(name: str, low: float, high: float):
    def parse(text: str) -> float:
        value = float(text)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"{name} must be between {low} and {high}, got {value}")
        return value
    return parse


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return value


def utc_datetime(text: str) -> datetime:
    """Parse an ISO 8601 time like 2026-10-19T05:00. A time with no offset is taken as UTC."""
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO 8601 time: {text!r} (try 2026-10-19T05:00)")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def time_zone_name(text: str) -> str:
    try:
        ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError):
        raise argparse.ArgumentTypeError(f"unknown time zone {text!r} (try America/New_York or UTC)")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orbittrack",
        description="Predict satellite passes over a ground station using fresh CelesTrak TLEs.",
    )

    sat = parser.add_argument_group("satellite")
    sat.add_argument("--sat", default="25544",
                     help="satellite name (e.g. \"ISS (ZARYA)\") or NORAD ID (default: 25544, the ISS)")
    sat.add_argument("--group", default="stations",
                     help="CelesTrak group to search, e.g. stations, weather, amateur (default: stations)")
    sat.add_argument("--tle-file", type=Path,
                     help="read TLEs from this local file instead of downloading from CelesTrak")

    station = parser.add_argument_group("ground station (default: West Lafayette, IN)")
    station.add_argument("--lat", type=bounded_float("latitude", -90, 90), help="latitude in degrees, north positive")
    station.add_argument("--lon", type=bounded_float("longitude", -180, 180), help="longitude in degrees, east positive")
    station.add_argument("--elev", type=float,
                         help="station height above the WGS84 ellipsoid in metres (default: 0)")
    station.add_argument("--tz", type=time_zone_name,
                         help="IANA time zone for displayed times (default: station's own, or UTC)")

    window = parser.add_argument_group("pass search")
    window.add_argument("--start", type=utc_datetime, help="start of the search window, ISO 8601 (default: now)")
    window.add_argument("--hours", type=positive_float, default=24.0, help="length of the search window (default: 24)")
    window.add_argument("--min-elevation", type=bounded_float("minimum elevation", 0, 89), default=10.0,
                        help="elevation mask in degrees (default: 10)")

    output = parser.add_argument_group("output")
    output.add_argument("--map", type=Path, metavar="FILE",
                        help="also save an interactive ground-track map of the search window to this HTML file")
    return parser


def station_from_args(args: argparse.Namespace) -> GroundStation:
    """Use West Lafayette unless the user gave a full custom location."""
    if args.lat is None and args.lon is None:
        station = WEST_LAFAYETTE
    else:
        station = GroundStation(
            name="Custom location",
            latitude_deg=args.lat,
            longitude_deg=args.lon,
            elevation_m=args.elev or 0.0,
        )
    # The station is frozen (immutable), so make a copy with the user's mask and time zone.
    return replace(station, min_elevation_deg=args.min_elevation, timezone=args.tz or station.timezone)


def lookup_satellite(satellites: list[EarthSatellite], query: str) -> EarthSatellite:
    """find_satellite(), but on a miss suggest names that contain the query (e.g. "ISS")."""
    try:
        return find_satellite(satellites, query)
    except LookupError:
        close = [s.name.strip() for s in satellites if query.strip().upper() in s.name.upper()]
        hint = f" Did you mean: {', '.join(close[:5])}?" if close else ""
        raise LookupError(f"no satellite matching {query!r} in this TLE set.{hint}")


# --- output ---

def print_current_position(satellite: EarthSatellite, station: GroundStation, t: Time) -> None:
    angles = look_angles(satellite, station, t)
    below = sub_point(satellite, t)

    if angles.altitude_deg < 0:
        status = "below the horizon"
    elif angles.altitude_deg < station.min_elevation_deg:
        status = f"above the horizon but below the {station.min_elevation_deg:g}° mask"
    else:
        status = "VISIBLE (above the elevation mask)"

    print(f"Satellite:  {satellite.name.strip()} (NORAD {satellite.model.satnum})")
    print(f"TLE epoch:  {satellite.epoch.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC "
          f"({tle_age_days(satellite, t):.2f} days from start)")
    print(f"Station:    {station.name} ({station.latitude_deg:.4f}, {station.longitude_deg:.4f}, "
          f"{station.elevation_m:.0f} m)")
    print(f"Time:       {t.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()
    print(f"Altitude:   {angles.altitude_deg:7.2f}°  -> {status}")
    print(f"Azimuth:    {angles.azimuth_deg:7.2f}°")
    print(f"Range:      {angles.range_km:7.0f} km")
    print()
    print(f"Sub-satellite point: {below.latitude_deg:.2f}, {below.longitude_deg:.2f} "
          f"at {below.height_km:.0f} km above Earth")


def print_passes(passes: list[Pass], station: GroundStation, hours: float) -> None:
    zone = station.local_zone()

    print(f"Passes above {station.min_elevation_deg:g}° in the next {hours:g} h "
          f"(times in {station.timezone}):")
    print()
    if not passes:
        print("  none")
        return

    print(f"  {'#':>2}  {'Date':<10}  {'Rise':>8}  {'Peak':>8}  {'Set':>8}  {'Max el':>6}  "
          f"{'Direction':<9}  {'Duration':>8}")
    for number, p in enumerate(passes, start=1):
        rise, peak, set_ = (moment.astimezone(zone) for moment in (p.rise, p.culmination, p.set))
        minutes, seconds = divmod(round(p.duration.total_seconds()), 60)
        direction = f"{compass_direction(p.rise_azimuth_deg)} -> {compass_direction(p.set_azimuth_deg)}"
        print(f"  {number:>2}  {rise:%Y-%m-%d}  {rise:%H:%M:%S}  {peak:%H:%M:%S}  {set_:%H:%M:%S}  "
              f"{p.max_elevation_deg:5.1f}°  {direction:<9}  {minutes:5d}m{seconds:02d}s")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat and --lon must be given together")
    if args.elev is not None and args.lat is None:
        parser.error("--elev needs --lat and --lon")

    station = station_from_args(args)
    try:
        tle_path = args.tle_file or fetch_tle_file(args.group)
        satellite = lookup_satellite(load_satellites(tle_path), args.sat)
    except (OSError, ValueError, LookupError) as error:
        parser.exit(1, f"error: {error}\n")

    ts = load.timescale()
    t = ts.from_datetime(args.start) if args.start else ts.now()

    passes = find_passes(satellite, station, t, args.hours)

    print_current_position(satellite, station, t)
    print()
    print_passes(passes, station, args.hours)

    if args.map:
        build_map(satellite, station, t, t + timedelta(hours=args.hours), passes).save(str(args.map))
        print()
        print(f"Map saved to {args.map.resolve()}")


if __name__ == "__main__":
    main()
