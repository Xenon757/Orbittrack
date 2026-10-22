"""Per-pass detail for working a satellite: antenna pointing and radio Doppler shift.

During a pass a ground station has to keep its antenna on the satellite (azimuth/elevation)
and keep its radio tuned: the satellite's motion toward or away from the station shifts the
frequency it hears (Doppler). Both come from the same thing, the satellite's position and
velocity relative to the station, sampled through the pass.
"""

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from skyfield.api import EarthSatellite, load

from orbittrack.passes import Pass
from orbittrack.station import GroundStation

SPEED_OF_LIGHT_KM_S = 299_792.458
# The ISS's worldwide amateur-radio voice downlink (ARISS), 145.800 MHz FM.
ISS_DOWNLINK_HZ = 145_800_000.0

_ts = load.timescale()


@dataclass(frozen=True)
class PassProfile:
    """One pass sampled at a fixed step from rise to set."""

    times: list[datetime]  # UTC
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    range_km: np.ndarray
    range_rate_km_s: np.ndarray  # negative = approaching, positive = receding
    frequency_hz: float  # the transmitted frequency
    doppler_hz: np.ndarray  # received minus transmitted frequency

    @property
    def received_hz(self) -> np.ndarray:
        """The frequency to tune the receiver to at each moment."""
        return self.frequency_hz + self.doppler_hz

    @property
    def azimuth_rate_deg_s(self) -> np.ndarray:
        """How fast the antenna must turn in azimuth between samples.

        Unwrapped so crossing north (359 -> 1 deg) counts as 2 deg, not 358. Near the zenith this
        gets very large: an overhead pass swings the azimuth through ~180 deg in seconds, which
        many hobby rotators (a few deg/s) can't follow. That's the "keyhole" problem.
        """
        unwrapped = np.unwrap(np.radians(self.azimuth_deg))
        seconds = np.array([(b - a).total_seconds() for a, b in zip(self.times, self.times[1:])])
        return np.degrees(np.abs(np.diff(unwrapped))) / seconds


def doppler_shift_hz(frequency_hz: float, range_rate_km_s: np.ndarray) -> np.ndarray:
    """Received minus transmitted frequency for a source moving at `range_rate` along the line of sight.

    First-order (non-relativistic) Doppler: f_received = f (1 - v/c). At satellite speeds the
    relativistic correction is a few parts in 10^10, far below what a radio can resolve.
    """
    return -frequency_hz * range_rate_km_s / SPEED_OF_LIGHT_KM_S


def pass_profile(
    satellite: EarthSatellite,
    station: GroundStation,
    p: Pass,
    frequency_hz: float = ISS_DOWNLINK_HZ,
    step_seconds: float = 1.0,
) -> PassProfile:
    """Sample a pass from rise to set: pointing angles, range, range rate and Doppler."""
    duration = p.duration.total_seconds()
    offsets = np.append(np.arange(0.0, duration, step_seconds), duration)
    times = _ts.from_datetime(p.rise) + offsets / 86400  # one Time holding every sample

    relative = (satellite - station.position()).at(times)
    altitude, azimuth, distance = relative.altaz()
    # Range rate is the relative velocity's component along the line of sight. The station's
    # velocity (Earth's rotation) is included because both positions are Earth-centered states.
    position = relative.position.km
    velocity = relative.velocity.km_per_s
    range_rate = np.sum(position * velocity, axis=0) / np.linalg.norm(position, axis=0)

    return PassProfile(
        times=[p.rise + timedelta(seconds=float(o)) for o in offsets],
        azimuth_deg=azimuth.degrees,
        elevation_deg=altitude.degrees,
        range_km=distance.km,
        range_rate_km_s=range_rate,
        frequency_hz=frequency_hz,
        doppler_hz=doppler_shift_hz(frequency_hz, range_rate),
    )


def pointing_csv(profile: PassProfile) -> str:
    """The pass as a time-stamped table an antenna controller and radio can follow."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["utc", "azimuth_deg", "elevation_deg", "range_km", "range_rate_km_s",
                     "doppler_hz", "receive_frequency_hz"])
    for i, moment in enumerate(profile.times):
        writer.writerow([
            moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            f"{profile.azimuth_deg[i]:.2f}",
            f"{profile.elevation_deg[i]:.2f}",
            f"{profile.range_km[i]:.1f}",
            f"{profile.range_rate_km_s[i]:.4f}",
            f"{profile.doppler_hz[i]:.0f}",
            f"{profile.received_hz[i]:.0f}",
        ])
    return out.getvalue()
