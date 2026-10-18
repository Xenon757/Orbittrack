"""Contact scheduling across several satellites and ground stations.

A "contact" is one pass of one satellite over one station. A station's antenna can only
point at one satellite at a time, so overlapping contacts at the same station conflict
and the schedule has to drop some of them.
"""

from dataclasses import dataclass
from datetime import datetime

from skyfield.api import EarthSatellite
from skyfield.timelib import Time

from orbittrack.passes import Pass, find_passes
from orbittrack.station import GroundStation


@dataclass(frozen=True)
class Contact:
    satellite: str
    station: str
    window: Pass  # rise/peak/set times and geometry for this satellite over this station

    @property
    def rise(self) -> datetime:
        return self.window.rise

    @property
    def set(self) -> datetime:
        return self.window.set


@dataclass(frozen=True)
class Conflict:
    dropped: Contact
    blocked_by: Contact  # the already-scheduled contact it overlapped at the same station


def find_contacts(
    satellites: list[EarthSatellite], stations: list[GroundStation], start: Time, hours: float
) -> list[Contact]:
    """Every pass of every satellite over every station, in order of rise time."""
    contacts = [
        Contact(satellite.name.strip(), station.name, p)
        for satellite in satellites
        for station in stations
        for p in find_passes(satellite, station, start, hours)
    ]
    return sorted(contacts, key=lambda c: c.rise)


def schedule_contacts(contacts: list[Contact]) -> tuple[list[Contact], list[Conflict]]:
    """Pick, for each station, the largest set of contacts that don't overlap.

    Uses the classic "earliest finish first" greedy rule: go through a station's contacts
    in order of set time and keep each one that starts after the last kept one ended.
    Finishing early leaves the most room for later contacts, and this is provably optimal
    for maximising the *number* of contacts. It ignores pass quality (e.g. max elevation)
    and mission priority, which a real scheduler would also weigh.
    """
    scheduled: list[Contact] = []
    conflicts: list[Conflict] = []

    for station in dict.fromkeys(c.station for c in contacts):  # unique, in first-seen order
        last_kept = None
        for contact in sorted((c for c in contacts if c.station == station), key=lambda c: c.set):
            if last_kept is None or contact.rise >= last_kept.set:
                scheduled.append(contact)
                last_kept = contact
            else:
                conflicts.append(Conflict(dropped=contact, blocked_by=last_kept))

    return sorted(scheduled, key=lambda c: c.rise), conflicts


def next_contact(scheduled: list[Contact], after: datetime) -> Contact | None:
    """The first scheduled contact that starts at or after `after`, if any."""
    return next((c for c in sorted(scheduled, key=lambda c: c.rise) if c.rise >= after), None)
