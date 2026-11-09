"""Simulation-only acceptance of a pot sliding on the machine's support surface."""

import math

import numpy as np


def supported_yaw_integral(previous, yaw_error_rad, *, maximum_deg=1.0):
    """Correct heading while keeping the support-constrained tilt integral fixed."""
    previous = np.asarray(previous, dtype=float)
    if (
        previous.shape != (3,)
        or not np.isfinite(previous).all()
        or not math.isfinite(yaw_error_rad)
    ):
        raise ValueError("invalid sliding orientation feedback")
    if not math.isfinite(maximum_deg) or not 0 < maximum_deg <= 5:
        raise ValueError("yaw correction limit must be finite and in (0, 5] degrees")
    updated = previous.copy()
    if abs(yaw_error_rad) > math.radians(0.25):
        updated[2] += 0.04 * yaw_error_rad
    updated[2] = np.clip(updated[2], -math.radians(maximum_deg), math.radians(maximum_deg))
    return updated


def supported_slide_settled(
    *,
    position_error_m,
    yaw_error_deg,
    tilt_deg,
    linear_speed_m_s,
    angular_speed_rad_s,
    upward_support_n,
    max_tilt_deg=2.0,
    max_yaw_error_deg=0.5,
):
    """Support permits bounded tilt and heading error, not drift or a missing grasp.

    Independent collision and grasp guards must still run. This function neither
    certifies a grasp nor declares the pot withdrawn from the machine.
    """
    if not math.isfinite(max_tilt_deg) or not 0 < max_tilt_deg <= 8:
        raise ValueError("supported tilt limit must be finite and in (0, 8] degrees")
    if not math.isfinite(max_yaw_error_deg) or not 0 < max_yaw_error_deg <= 10:
        raise ValueError("supported heading tolerance must be finite and in (0, 10] degrees")
    values = (
        position_error_m,
        yaw_error_deg,
        tilt_deg,
        linear_speed_m_s,
        angular_speed_rad_s,
        upward_support_n,
    )
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("need finite nonnegative sliding measurements")
    return (
        upward_support_n > 0.02
        and position_error_m < 0.002
        and yaw_error_deg < max_yaw_error_deg
        and tilt_deg < max_tilt_deg
        and linear_speed_m_s < 0.003
        and angular_speed_rad_s < 0.03
    )
