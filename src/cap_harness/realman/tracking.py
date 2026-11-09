"""Bounded simulation object-tracking correction; no robot IO."""

import numpy as np


def bumpless_rotation_limit(previous, *, requested_deg):
    """Avoid deleting stored compensation when changing controller phases.

    If the previous correction exceeds the requested bound, preserve its norm
    until ordinary error feedback reduces it. This does not alter independent
    grasp, collision, actuator or load limits.
    """
    previous = np.asarray(previous, dtype=float)
    if (
        previous.shape != (3,)
        or not np.isfinite(previous).all()
        or not np.isfinite(requested_deg)
        or requested_deg <= 0
    ):
        raise ValueError("invalid correction transition")
    return max(float(requested_deg), float(np.rad2deg(np.linalg.norm(previous))))


def bounded_rotation_integral(
    previous, error_rotvec, *, gain=0.04, deadband_deg=0.5, maximum_deg=1.0
):
    previous, error = np.asarray(previous, dtype=float), np.asarray(error_rotvec, dtype=float)
    if (
        previous.shape != (3,)
        or error.shape != (3,)
        or not np.isfinite(previous).all()
        or not np.isfinite(error).all()
        or not np.isfinite([gain, deadband_deg, maximum_deg]).all()
        or not 0 < gain <= 1
        or not 0 < deadband_deg < maximum_deg
    ):
        raise ValueError("invalid angular integral parameters")
    updated = previous.copy()
    if np.linalg.norm(error) > np.deg2rad(deadband_deg):
        updated += gain * error
    updated *= min(1.0, np.deg2rad(maximum_deg) / max(np.linalg.norm(updated), 1e-12))
    return updated
