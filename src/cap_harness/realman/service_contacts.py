"""Extend the single-cup baseline pot guard to every service cup and the rack."""

from .contact_guards import prohibited_pot_contacts as _baseline_pot_contacts


def prohibited_pot_contacts(contacts, *, force_threshold_n=0.02):
    contacts = list(contacts)
    failures = _baseline_pot_contacts(contacts, force_threshold_n=force_threshold_n)
    for contact in contacts:
        bodies = [contact["body1"], contact["body2"]]
        if "pot" not in bodies or contact["normal_force_n"] <= force_threshold_n:
            continue
        other = bodies[1 - bodies.index("pot")]
        is_cup = other.startswith("cup_") and other[4:].isdigit()
        is_rack = other == "cup_dispenser" or other.startswith("retention_finger")
        if (is_cup or is_rack) and contact not in failures:
            failures.append(dict(contact))
    return failures
