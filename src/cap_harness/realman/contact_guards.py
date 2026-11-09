"""Simulation contact audits. Examine every contact, independent of ordering."""

import math

from .platform import is_machine_support_geom


def prohibited_pot_contacts(contacts, *, force_threshold_n=0.02):
    """Filter normalized contact records; intended hotplate support is allowed.

    Each record supplies body1/body2, geom1/geom2 and normal_force_n. The spring
    outlet is a separate compliant body, not the fixed coffee-machine housing.
    Robot/tool contacts remain subject to the independent robot/grasp guards.
    """
    if not math.isfinite(force_threshold_n) or force_threshold_n < 0:
        raise ValueError("need a finite nonnegative force threshold")
    failures = []
    for contact in contacts:
        bodies = [contact["body1"], contact["body2"]]
        names = [contact["geom1"], contact["geom2"]]
        force = contact["normal_force_n"]
        if not math.isfinite(force):
            raise ValueError("nonfinite contact force")
        if "pot" not in bodies or force <= force_threshold_n:
            continue
        other = 1 - bodies.index("pot")
        forbidden = bodies[other] in {"cup_0", "table"} or (
            bodies[other] == "coffee_machine" and not is_machine_support_geom(names[other])
        )
        if forbidden:
            failures.append(dict(contact))
    return failures
