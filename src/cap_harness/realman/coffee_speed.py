"""Evidence-based transport speed selection for the coffee workflow."""

from __future__ import annotations

import math


def pot_transport_speed_factor(
    free_speed_factor: float,
    contact_speed_factor: float,
    *,
    independently_clear: bool,
    upward_support_n: float,
    support_tolerance_n: float = 0.02,
) -> float:
    """Select free transport speed only after independent clearance is proven.

    A held pot may move at the free-space factor after it has cleared the
    machine and is not being carried by upward platform support. Otherwise,
    retain the contact speed factor.
    """
    values = (free_speed_factor, contact_speed_factor, upward_support_n, support_tolerance_n)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("speed factors and support values must be numeric, not boolean or string")
    free = float(free_speed_factor)
    contact = float(contact_speed_factor)
    support = float(upward_support_n)
    tolerance = float(support_tolerance_n)
    if not math.isfinite(free) or not 1 <= free <= 4:
        raise ValueError("free speed factor must be finite and in [1, 4]")
    if not math.isfinite(contact) or not 1 <= contact <= 2:
        raise ValueError("contact speed factor must be finite and in [1, 2]")
    if not math.isfinite(support) or support < 0:
        raise ValueError("upward support must be finite and nonnegative")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("support tolerance must be finite and nonnegative")
    if not isinstance(independently_clear, bool):
        raise ValueError("independent clearance must be a boolean predicate")
    return free if independently_clear and support <= tolerance else contact
