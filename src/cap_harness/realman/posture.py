"""Geometric upright constraint for frozen simulation fixtures, not motion limits."""

from __future__ import annotations

import math


def upper_body_posture(model, *, maximum_tilt_deg=None):
    """Measure chest +Z against world +Z after composing the full body chain.

    This checks the generated fixture, not stability, reachability, hardware
    calibration or a trajectory between body configurations. URDF joint limits
    remain separate and are never widened here.
    """
    import mujoco
    import numpy as np

    if maximum_tilt_deg is not None:
        if isinstance(maximum_tilt_deg, bool):
            raise ValueError("maximum upper-body tilt must be numeric")
        maximum_tilt_deg = float(maximum_tilt_deg)
        if not math.isfinite(maximum_tilt_deg) or not 0 <= maximum_tilt_deg <= 90:
            raise ValueError("maximum upper-body tilt must be in [0, 90] degrees")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    body = data.body("xb_link")
    tilt = float(np.rad2deg(np.arccos(np.clip(body.xmat.reshape(3, 3)[2, 2], -1, 1))))
    if maximum_tilt_deg is not None and tilt > maximum_tilt_deg + 1e-6:
        raise ValueError(f"upper-body tilt {tilt:.6f} deg exceeds {maximum_tilt_deg:.6f} deg")
    return {
        "reference_body": "xb_link",
        "upright_axis": "body +Z relative to world +Z",
        "tilt_deg": tilt,
        "maximum_tilt_deg": maximum_tilt_deg,
        "origin_world_m": body.xpos.tolist(),
        "simulation_fixture_only": True,
    }
