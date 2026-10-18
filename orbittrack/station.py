"""Ground station definition: where the antenna is and how low it can usefully look."""

from dataclasses import dataclass
from zoneinfo import ZoneInfo

from skyfield.api import wgs84
from skyfield.toposlib import GeographicPosition


@dataclass(frozen=True)
class GroundStation:
    name: str
    latitude_deg: float  # north positive
    longitude_deg: float  # east positive, so the western hemisphere is negative
    elevation_m: float  # height above the WGS84 ellipsoid
    # Below ~10 degrees the signal passes through more atmosphere and buildings/trees
    # block the view, so passes lower than this are usually not worth scheduling.
    min_elevation_deg: float = 10.0
    # IANA zone name, used only to display times; all calculations stay in UTC.
    timezone: str = "UTC"

    def local_zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def position(self) -> GeographicPosition:
        """The station as a point on the WGS84 ellipsoid, which Skyfield uses for topocentric math."""
        return wgs84.latlon(self.latitude_deg, self.longitude_deg, elevation_m=self.elevation_m)


# Purdue University campus, West Lafayette, Indiana.
WEST_LAFAYETTE = GroundStation(
    name="West Lafayette, IN",
    latitude_deg=40.4237,
    longitude_deg=-86.9212,
    elevation_m=187.0,
    timezone="America/Indiana/Indianapolis",
)
