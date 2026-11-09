"""Offline pivot-point fit; contact evidence and reference geometry are external.

The fitted point is the installed probe tip. It is NOT automatically the grasp
center, and a pivot fit does not identify the tool's orientation axes.
"""

import numpy as np

from .handeye_solver import excitation


def fit_pivot(poses):
    if len(poses) < 6:
        raise ValueError("requires at least six independently aligned stationary poses")
    e = excitation(poses)
    if e["maximum_rotation_deg"] < 15 or e["second_to_first_ratio"] < 0.2:
        raise ValueError("pivot orientations insufficiently diverse")
    a = np.vstack([np.column_stack([p[:3, :3], -np.eye(3)]) for p in poses])
    b = np.concatenate([-p[:3, 3] for p in poses])
    solution, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    if rank != 6 or singular[0] / singular[-1] > 1000:
        raise ValueError("pivot fit underconstrained")
    return {
        "probe_tip_in_tool_m": solution[:3].tolist(),
        "fixed_point_in_base_m": solution[3:].tolist(),
        "condition_number": float(singular[0] / singular[-1]),
        "excitation": e,
        "grasp_center_calibrated": False,
        "tool_orientation_calibrated": False,
        "contact_verified_by_solver": False,
    }


def evaluate_pivot(poses, result):
    tip = np.asarray(result["probe_tip_in_tool_m"])
    fixed = np.asarray(result["fixed_point_in_base_m"])
    errors = np.asarray([np.linalg.norm(p[:3, :3] @ tip + p[:3, 3] - fixed) * 1000 for p in poses])
    return {
        "count": len(poses),
        "rms_mm": float(np.sqrt(np.mean(errors**2))),
        "max_mm": float(errors.max()),
        "per_pose_mm": errors.tolist(),
    }
