"""Contact scheduling: conflict rules on synthetic contacts, plus real multi-satellite data."""

import itertools
import random
from datetime import datetime, timedelta, timezone

from skyfield.api import load

from orbittrack.passes import Pass, find_passes
from orbittrack.schedule import Contact, find_contacts, next_contact, schedule_contacts
from orbittrack.station import WEST_LAFAYETTE, GroundStation
from orbittrack.tle import find_satellite, load_satellites

ts = load.timescale()
BASE = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
START = ts.from_datetime(BASE)


def contact(satellite: str, start_min: float, end_min: float, station: str = "A") -> Contact:
    """A made-up contact from start_min to end_min minutes after BASE."""
    rise, set_ = BASE + timedelta(minutes=start_min), BASE + timedelta(minutes=end_min)
    window = Pass(rise, rise + (set_ - rise) / 2, set_, 45.0, 0.0, 180.0)
    return Contact(satellite, station, window)


def names(contacts):
    return [c.satellite for c in contacts]


# --- conflict rules ---

def test_non_overlapping_contacts_are_all_scheduled():
    contacts = [contact("a", 0, 5), contact("b", 10, 15), contact("c", 20, 25)]
    scheduled, conflicts = schedule_contacts(contacts)
    assert names(scheduled) == ["a", "b", "c"]
    assert conflicts == []


def test_overlapping_contacts_at_one_station_conflict():
    scheduled, conflicts = schedule_contacts([contact("a", 0, 10), contact("b", 5, 15)])
    assert names(scheduled) == ["a"]
    assert conflicts[0].dropped.satellite == "b"
    assert conflicts[0].blocked_by.satellite == "a"


def test_back_to_back_contacts_do_not_conflict():
    # One ends exactly as the next begins; the antenna can (ideally) switch instantly.
    scheduled, conflicts = schedule_contacts([contact("a", 0, 10), contact("b", 10, 20)])
    assert names(scheduled) == ["a", "b"] and conflicts == []


def test_overlaps_at_different_stations_do_not_conflict():
    scheduled, conflicts = schedule_contacts([contact("a", 0, 10, "A"), contact("b", 5, 15, "B")])
    assert len(scheduled) == 2 and conflicts == []


def test_earliest_finish_beats_taking_the_first_contact_to_start():
    # Taking the long contact that starts first would block both short ones.
    contacts = [contact("long", 0, 100), contact("short1", 10, 20), contact("short2", 30, 40)]
    scheduled, conflicts = schedule_contacts(contacts)
    assert names(scheduled) == ["short1", "short2"]
    assert [c.dropped.satellite for c in conflicts] == ["long"]


def test_greedy_schedules_as_many_contacts_as_the_best_possible_choice():
    # Compare against brute force (try every subset) on many small random cases.
    rng = random.Random(42)
    for _ in range(200):
        contacts = []
        for i in range(7):
            start = rng.uniform(0, 60)
            contacts.append(contact(f"s{i}", start, start + rng.uniform(1, 15)))

        def no_overlaps(subset):
            ordered = sorted(subset, key=lambda c: c.rise)
            return all(a.set <= b.rise for a, b in zip(ordered, ordered[1:]))

        best = max(len(s) for r in range(len(contacts) + 1)
                   for s in itertools.combinations(contacts, r) if no_overlaps(s))
        scheduled, conflicts = schedule_contacts(contacts)
        assert no_overlaps(scheduled)
        assert len(scheduled) == best
        assert len(scheduled) + len(conflicts) == len(contacts)


def test_next_contact_is_the_first_one_at_or_after_the_given_time():
    scheduled = [contact("b", 30, 35), contact("a", 10, 15)]
    assert next_contact(scheduled, BASE).satellite == "a"
    assert next_contact(scheduled, BASE + timedelta(minutes=20)).satellite == "b"
    assert next_contact(scheduled, BASE + timedelta(minutes=40)) is None


# --- real satellites from the fixture TLE ---

def test_find_contacts_covers_every_satellite_station_pair_in_time_order(tle_path):
    satellites = load_satellites(tle_path)
    chosen = [find_satellite(satellites, "25544"), find_satellite(satellites, "CSS (TIANHE)")]
    houston = GroundStation("Houston, TX", 29.5593, -95.0900, 10.0)
    contacts = find_contacts(chosen, [WEST_LAFAYETTE, houston], START, 24)

    expected = sum(len(find_passes(s, st, START, 24)) for s in chosen for st in (WEST_LAFAYETTE, houston))
    assert len(contacts) == expected > 0
    assert contacts == sorted(contacts, key=lambda c: c.rise)
    assert {c.station for c in contacts} == {"West Lafayette, IN", "Houston, TX"}


def test_docked_cargo_ship_always_conflicts_with_its_station(tle_path):
    # Tianzhou-10 is docked to the Chinese station, so their passes are identical and the
    # antenna can only take one of each pair.
    satellites = load_satellites(tle_path)
    chosen = [find_satellite(satellites, "CSS (TIANHE)"), find_satellite(satellites, "TIANZHOU-10")]
    scheduled, conflicts = schedule_contacts(find_contacts(chosen, [WEST_LAFAYETTE], START, 24))

    assert len(scheduled) == len(conflicts) > 0
    assert all(abs(c.dropped.rise - c.blocked_by.rise) < timedelta(seconds=5) for c in conflicts)


def test_fairbanks_never_sees_the_iss_above_ten_degrees(iss):
    # The ISS's ground track tops out at 51.6 deg; with a 10 deg mask the coverage radius is
    # ~12.5 deg, so it's never above the mask north of ~64.1 deg. Fairbanks is at 64.8 N.
    fairbanks = GroundStation("Fairbanks, AK", 64.8378, -147.7164, 136.0)
    assert find_contacts([iss], [fairbanks], START, 72) == []
