"""Where a satellite is right now, as seen from a ground station."""

from typing import NamedTuple

from skyfield.api import EarthSatellite, wgs84
from skyfield.timelib import Time

from orbittrack.station import GroundStation


class LookAngles(NamedTuple):
    altitude_deg: float  # angle above the horizon; negative means below it
    azimuth_deg: float  # compass direction: 0 = north, 90 = east, 180 = south, 270 = west
    range_km: float  # straight-line distance from the antenna to the satellite


class SubPoint(NamedTuple):
    latitude_deg: float
    longitude_deg: float
    height_km: float  # height above the WGS84 ellipsoid (what people usually mean by "orbit altitude")


def look_angles(satellite: EarthSatellite, station: GroundStation, t: Time) -> LookAngles:
    """Topocentric altitude/azimuth/range: where to point the antenna at time `t`.

    Subtracting the station's position from the satellite's position (both Earth-centered)
    gives the vector from antenna to satellite; Skyfield then rotates it into the station's
    local horizon frame (up / north / east) to get the angles.
    """
    difference = satellite - station.position()
    altitude, azimuth, distance = difference.at(t).altaz()
    return LookAngles(altitude.degrees, azimuth.degrees, distance.km)


def sub_point(satellite: EarthSatellite, t: Time) -> SubPoint:
    """The point on Earth directly below the satellite, and how high above it the satellite is."""
    geocentric = satellite.at(t)
    lat, lon = wgs84.latlon_of(geocentric)
    height = wgs84.height_of(geocentric)
    return SubPoint(lat.degrees, lon.degrees, height.km)
