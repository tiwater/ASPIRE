"""Candidate pot-centre geometry; does not execute or certify a robot motion."""

import numpy as np


def clearance_entry_centres(
    seat_position_m,
    outward_heading_rad,
    remaining_distances_m,
    *,
    detour_span_m,
    lateral_offset_m,
):
    """Return a planar entrance detour with exact nominal start/end centres.

    Positive lateral offset follows (-sin(heading), cos(heading)), perpendicular
    to the outward handle axis. The offset is sinusoidal over the last
    ``detour_span_m`` of insertion and zero outside it. Interpolation between
    returned centres is the caller's responsibility; wrist orientation is not
    changed here. Requires separate robot, object-clearance and contact checks.
    """
    seat = np.asarray(seat_position_m, dtype=float)
    distances = np.asarray(remaining_distances_m, dtype=float)
    parameters = np.array([outward_heading_rad, detour_span_m, lateral_offset_m])
    if (
        seat.shape != (3,)
        or not np.isfinite(seat).all()
        or distances.ndim != 1
        or len(distances) < 2
        or not np.isfinite(distances).all()
        or np.any(distances < 0)
        or np.any(np.diff(distances) >= 0)
        or distances[-1] != 0
        or parameters.shape != (3,)
        or not np.isfinite(parameters).all()
        or detour_span_m <= 0
    ):
        raise ValueError(
            "need finite planar entry parameters and decreasing distances ending at zero"
        )
    axis = np.array([np.cos(outward_heading_rad), np.sin(outward_heading_rad), 0.0])
    normal = np.array([-axis[1], axis[0], 0.0])
    bump = np.zeros_like(distances)
    inside = (distances > 0) & (distances < detour_span_m)
    bump[inside] = lateral_offset_m * np.sin(np.pi * distances[inside] / detour_span_m)
    return seat + distances[:, None] * axis + bump[:, None] * normal
