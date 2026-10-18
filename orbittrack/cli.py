"""Command-line entry point.

Milestone 1: print where the ISS is right now as seen from West Lafayette.
(Satellite, location and time-window options arrive in Milestone 3.)
"""

from skyfield.api import load

from orbittrack.observe import look_angles, sub_point
from orbittrack.station import WEST_LAFAYETTE
from orbittrack.tle import fetch_tle_file, find_satellite, load_satellites, tle_age_days

ISS_NORAD_ID = "25544"


def main() -> None:
    station = WEST_LAFAYETTE
    satellites = load_satellites(fetch_tle_file("stations"))
    iss = find_satellite(satellites, ISS_NORAD_ID)

    t = load.timescale().now()
    angles = look_angles(iss, station, t)
    below = sub_point(iss, t)

    if angles.altitude_deg < 0:
        status = "below the horizon"
    elif angles.altitude_deg < station.min_elevation_deg:
        status = f"above the horizon but below the {station.min_elevation_deg:.0f} deg mask"
    else:
        status = "VISIBLE (above the elevation mask)"

    print(f"Satellite:  {iss.name} (NORAD {iss.model.satnum})")
    print(f"TLE epoch:  {iss.epoch.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC "
          f"({tle_age_days(iss, t):.2f} days old)")
    print(f"Station:    {station.name} ({station.latitude_deg:.4f}, {station.longitude_deg:.4f})")
    print(f"Time:       {t.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print()
    print(f"Altitude:   {angles.altitude_deg:7.2f} deg  -> {status}")
    print(f"Azimuth:    {angles.azimuth_deg:7.2f} deg")
    print(f"Range:      {angles.range_km:7.0f} km")
    print()
    print(f"Sub-satellite point: {below.latitude_deg:.2f}, {below.longitude_deg:.2f} "
          f"at {below.height_km:.0f} km above Earth")


if __name__ == "__main__":
    main()
