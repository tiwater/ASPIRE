"""Eligibility for the geometry-qualified coffee release recovery in simulation.

The caller still has to execute and verify release. This decision never converts
a supported pot or a small contact force into evidence of tool separation.
"""
from __future__ import annotations

import math


RELEASE_MODEL_SHA256 = "990e7fff88bde142346b151a77f08bedc47f4b47c013c7339b735bbf17080917"
RETRY_OPENING = .1975
REQUIRED_SUPPORT = (
    "pot_on_hotplate", "lid_outlet_aligned", "pot_upright",
    "right_gripper_empty", "previous_served_cups_preserved",
    "rack_inventory_supported",
)


def release_retry_allowed(predicates, contacts, *, model_sha256, opening):
    """Permit one known support-edge recovery, with unchanged release criteria.

    The exact mesh pair and model identity deliberately limit this recovery to
    the tested scene. A remodel or another contact needs separate qualification.
    Contacts use the rollout's geom1/geom2/normal_force_n records.
    """
    if model_sha256 != RELEASE_MODEL_SHA256 or opening != .205:
        return False
    if any(predicates.get(name) is not True for name in REQUIRED_SUPPORT):
        return False
    if predicates.get("pot_stable_after_release") is not False:
        return False
    if predicates.get("left_gripper_released") is not False:
        return False
    residual = []
    for contact in contacts:
        names = (contact["geom1"], contact["geom2"])
        if not any(n.startswith(("ltool_", "left_left_pad", "left_right_pad")) for n in names):
            continue
        force = contact["normal_force_n"]
        if not isinstance(force, (int, float)) or not math.isfinite(force) or force < 0:
            return False
        if force <= .001:
            continue
        if force > .02 or set(names) != {
            "ltool_Right_Support_Link_collision_0", "handle_collision_12"
        }:
            return False
        residual.append(contact)
    return bool(residual)
