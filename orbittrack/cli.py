"""Command-line entry point.

Milestones 1-2: print where the ISS is right now as seen from West Lafayette, then
every pass above the elevation mask in the next 24 hours.
(Satellite, location and time-window options arrive in Milestone 3.)
"""

from skyfield.api import EarthSatellite, load
from skyfield.timelib import Time

from orbittrack.observe import look_angles, sub_point
from orbittrack.passes import compass_direction, find_passes
from orbittrack.station import WEST_LAFAYETTE, GroundStation
from orbittrack.tle import fetch_tle_file, find_satellite, load_satellites, tle_age_days

ISS_NORAD_ID = "25544"
WINDOW_HOURS = 24.0


def print_current_position(satellite: EarthSatellite, station: GroundStation, t: Time) -> None:
    angles = look_angles(satellite, station, t)
    below = sub_point(satellite, t)

    if angles.altitude_deg < 0:
        status = "below the horizon"
    elif angles.altitude_deg < station.min_elevation_deg:
        status = f"above the horizon but below the {station.min_elevation_deg:.0f} deg mask"
    else:
        status = "VISIBLE (above the elevation mask)"

    print(f"Satellite:  {satellite.name} (NORAD {satellite.model.satnum})")
    print(f"TLE epoch:  {satellite.epoch.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC "
          f"({tle_age_days(satellite, t):.2f} days old)")
    print(f"Station:    {station.name} ({station.latitude_deg:.4f}, {station.longitude_deg:.4f})")
    print(f"Time:       {t.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()
    print(f"Altitude:   {angles.altitude_deg:7.2f} deg  -> {status}")
    print(f"Azimuth:    {angles.azimuth_deg:7.2f} deg")
    print(f"Range:      {angles.range_km:7.0f} km")
    print()
    print(f"Sub-satellite point: {below.latitude_deg:.2f}, {below.longitude_deg:.2f} "
          f"at {below.height_km:.0f} km above Earth")


def print_passes(satellite: EarthSatellite, station: GroundStation, t: Time) -> None:
    passes = find_passes(satellite, station, t, WINDOW_HOURS)
    zone = station.local_zone()

    print(f"Passes above {station.min_elevation_deg:.0f}° in the next {WINDOW_HOURS:.0f} h "
          f"(times in {station.timezone}):")
    print()
    if not passes:
        print("  none")
        return

    print(f"  {'Date':<10}  {'Rise':>8}  {'Peak':>8}  {'Set':>8}  {'Max el':>6}  {'Direction':<9}  {'Duration':>8}")
    for p in passes:
        rise, peak, set_ = (moment.astimezone(zone) for moment in (p.rise, p.culmination, p.set))
        minutes, seconds = divmod(round(p.duration.total_seconds()), 60)
        direction = f"{compass_direction(p.rise_azimuth_deg)} -> {compass_direction(p.set_azimuth_deg)}"
        print(f"  {rise:%Y-%m-%d}  {rise:%H:%M:%S}  {peak:%H:%M:%S}  {set_:%H:%M:%S}  "
              f"{p.max_elevation_deg:5.1f}°  {direction:<9}  {minutes:5d}m{seconds:02d}s")


def main() -> None:
    station = WEST_LAFAYETTE
    satellites = load_satellites(fetch_tle_file("stations"))
    iss = find_satellite(satellites, ISS_NORAD_ID)
    t = load.timescale().now()

    print_current_position(iss, station, t)
    print()
    print_passes(iss, station, t)


if __name__ == "__main__":
    main()
