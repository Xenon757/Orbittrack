"""How accurate are our SGP4/TLE predictions? Compare them with NASA's ISS trajectory.

NASA's Johnson Space Center (the TOPO office in Flight Operations) publishes the ISS ephemeris
in the CCSDS Orbit Ephemeris Message (OEM) format: Earth-centered positions and velocities every
4 minutes for about 15 days, built from tracking data and high-fidelity force models.

    https://nasa-public-data.s3.amazonaws.com/iss-coords/current/ISS_OEM/ISS.OEM_J2K_EPH.txt

Read the results in two parts:
- At the start of the OEM, NASA's state comes from recent tracking, so the difference there is
  a measurement of the TLE's error.
- Later rows are NASA's *forecast*. Differences there are SGP4 and NASA's high-fidelity model
  diverging (mostly over future drag), which includes NASA's own forecast uncertainty.

Differences are split into radial / along-track / cross-track components. They're almost entirely
along-track: the satellite is in the right place on its orbit, but early or late, because drag
is what a TLE models least well.

Usage:
    python -m orbittrack.accuracy --oem ISS.OEM_J2K_EPH.txt
    python -m orbittrack.accuracy --oem ISS.OEM_J2K_EPH.txt --tle-file tests/fixtures/stations_2026-10-18.tle
"""

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from skyfield.api import EarthSatellite, load

from orbittrack.tle import fetch_tle_file, load_satellites, lookup_satellite

ISS_NORAD_ID = "25544"
DATA_LINE = re.compile(r"^\d{4}-\d{2}-\d{2}T\S+(\s+[-+0-9.Ee]+){6}\s*$")


@dataclass(frozen=True)
class Ephemeris:
    """State vectors from an OEM file: UTC times, positions (km) and velocities (km/s)."""

    times: list[datetime]
    positions_km: np.ndarray  # shape (n, 3)
    velocities_km_s: np.ndarray  # shape (n, 3)
    reference_frame: str
    time_system: str


def parse_oem(text: str) -> Ephemeris:
    """Parse a CCSDS OEM text file (KVN format).

    Keeps every data line in every segment and skips metadata, comments and any covariance
    blocks. Only Earth-centered EME2000 data in UTC is accepted: that's what NASA publishes for
    the ISS, and it lines up with Skyfield's Earth-centered frame (they differ by the ~0.02
    arcsecond "frame bias", under a metre at the ISS's distance).
    """
    meta: dict[str, str] = {}
    times, states = [], []
    in_covariance = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("COVARIANCE_START"):
            in_covariance = True
        elif line.startswith("COVARIANCE_STOP"):
            in_covariance = False
        elif in_covariance or not line or line.startswith("COMMENT"):
            continue
        elif "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            meta[key] = value
        elif DATA_LINE.match(line):
            fields = line.split()
            times.append(datetime.fromisoformat(fields[0]).replace(tzinfo=timezone.utc))
            states.append([float(value) for value in fields[1:7]])

    frame, time_system, center = meta.get("REF_FRAME"), meta.get("TIME_SYSTEM"), meta.get("CENTER_NAME")
    if (frame, time_system, center) != ("EME2000", "UTC", "Earth"):
        raise ValueError(f"expected Earth-centered EME2000 data in UTC, got {center}/{frame}/{time_system}")
    if not states:
        raise ValueError("no state vectors found in OEM file")

    array = np.array(states)
    return Ephemeris(times, array[:, :3], array[:, 3:], reference_frame="EME2000", time_system="UTC")


@dataclass(frozen=True)
class Comparison:
    """SGP4-minus-NASA position differences at each OEM time, in km."""

    times: list[datetime]
    tle_age_days: np.ndarray
    total_km: np.ndarray
    along_km: np.ndarray  # + means SGP4 is ahead of NASA along the orbit
    cross_km: np.ndarray  # + means SGP4 is on the orbit-normal (angular momentum) side
    radial_km: np.ndarray  # + means SGP4 is higher
    speed_km_s: np.ndarray

    @property
    def timing_error_s(self) -> np.ndarray:
        """Along-track error expressed as time: how early or late the satellite arrives."""
        return self.along_km / self.speed_km_s


def radial_along_cross(position_km: np.ndarray, velocity_km_s: np.ndarray, difference_km: np.ndarray):
    """Project position differences onto the reference orbit's local frame.

    Radial points away from Earth's center, cross-track along the orbit normal (r x v), and
    along-track completes the right-handed set (roughly the direction of travel).
    """
    radial = position_km / np.linalg.norm(position_km, axis=1, keepdims=True)
    cross = np.cross(position_km, velocity_km_s)
    cross /= np.linalg.norm(cross, axis=1, keepdims=True)
    along = np.cross(cross, radial)
    dot = lambda axis: np.sum(difference_km * axis, axis=1)
    return dot(along), dot(cross), dot(radial)


def compare(satellite: EarthSatellite, ephemeris: Ephemeris) -> Comparison:
    """Propagate the TLE with SGP4 to every OEM time and measure the difference."""
    ts = load.timescale()
    times = ts.from_datetimes(ephemeris.times)
    sgp4_km = satellite.at(times).position.km.T  # Earth-centered, same axes as EME2000 (see parse_oem)
    difference = sgp4_km - ephemeris.positions_km
    along, cross, radial = radial_along_cross(ephemeris.positions_km, ephemeris.velocities_km_s, difference)
    return Comparison(
        times=ephemeris.times,
        tle_age_days=np.asarray(times - satellite.epoch),
        total_km=np.linalg.norm(difference, axis=1),
        along_km=along,
        cross_km=cross,
        radial_km=radial,
        speed_km_s=np.linalg.norm(ephemeris.velocities_km_s, axis=1),
    )


def summary_rows(result: Comparison, days_ahead: tuple[float, ...] = (0, 1, 2, 3, 5, 7, 10, 14)) -> list[dict]:
    """The comparison at a few checkpoints after the start of the OEM."""
    start = result.times[0]
    elapsed_days = np.array([(t - start).total_seconds() / 86400 for t in result.times])
    rows = []
    for day in days_ahead:
        if day > elapsed_days[-1]:
            break
        i = int(np.argmin(np.abs(elapsed_days - day)))
        rows.append({
            "time": result.times[i],
            "tle_age_days": float(result.tle_age_days[i]),
            "total_km": float(result.total_km[i]),
            "along_km": float(result.along_km[i]),
            "cross_km": float(result.cross_km[i]),
            "radial_km": float(result.radial_km[i]),
            "timing_s": float(result.timing_error_s[i]),
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="orbittrack.accuracy", description=__doc__.split("\n\n")[0])
    parser.add_argument("--oem", type=Path, required=True, help="NASA ISS OEM file (ISS.OEM_J2K_EPH.txt)")
    parser.add_argument("--tle-file", type=Path, help="TLE file to test (default: download the current stations group)")
    parser.add_argument("--csv", type=Path, help="also write every sample's differences to this CSV file")
    args = parser.parse_args(argv)

    ephemeris = parse_oem(args.oem.read_text())
    iss = lookup_satellite(load_satellites(args.tle_file or fetch_tle_file("stations")), ISS_NORAD_ID)
    result = compare(iss, ephemeris)

    print(f"TLE epoch:  {iss.epoch.utc_strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"NASA OEM:   {ephemeris.times[0]:%Y-%m-%d %H:%M} to {ephemeris.times[-1]:%Y-%m-%d %H:%M} UTC, "
          f"{len(ephemeris.times)} state vectors ({ephemeris.reference_frame}, {ephemeris.time_system})")
    print()
    print(f"Measured at the OEM start (tracking-based): {result.total_km[0]:.2f} km "
          f"({result.timing_error_s[0]:+.2f} s along-track) with a {result.tle_age_days[0]:.2f}-day-old TLE")
    print()
    print("SGP4 minus NASA (km). Along-track + = SGP4 ahead; timing = along-track error / speed.")
    print("Rows after the first compare against NASA's forecast, so they show divergence, not pure TLE error.")
    print(f"  {'UTC':<16}  {'TLE age':>8}  {'Total':>8}  {'Along':>8}  {'Cross':>7}  {'Radial':>7}  {'Timing':>8}")
    for row in summary_rows(result):
        print(f"  {row['time']:%Y-%m-%d %H:%M}  {row['tle_age_days']:6.2f} d  {row['total_km']:8.2f}  "
              f"{row['along_km']:8.2f}  {row['cross_km']:7.2f}  {row['radial_km']:7.2f}  {row['timing_s']:+7.2f} s")
    share = np.sum(result.along_km ** 2) / np.sum(result.total_km ** 2)
    print()
    print(f"Along-track share of the squared error over the whole file: {share:.1%}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["utc", "tle_age_days", "total_km", "along_km", "cross_km", "radial_km", "timing_s"])
            for i, moment in enumerate(result.times):
                writer.writerow([moment.isoformat(), f"{result.tle_age_days[i]:.4f}", f"{result.total_km[i]:.3f}",
                                 f"{result.along_km[i]:.3f}", f"{result.cross_km[i]:.3f}",
                                 f"{result.radial_km[i]:.3f}", f"{result.timing_error_s[i]:.3f}"])
        print(f"Wrote {len(result.times)} rows to {args.csv.resolve()}")


if __name__ == "__main__":
    main()
