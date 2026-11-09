"""Offline, board-referenced TCP geometry for a potentially moving robot base.

T_A_B maps B coordinates into A. All translations are metres. Inputs must come
from independently qualified, time-matched observations of the SAME board.
These algebra routines do not verify contact, timestamps, board identity,
camera mounting, or collision clearance, and never actuate a robot.
"""

import numpy as np

from .tcp_calibration import evaluate_pivot, fit_pivot


def _rigid(value):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("requires a finite 4x4 transform")
    r = t[:3, :3]
    if not (
        np.allclose(t[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        and np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0)
        and np.isclose(np.linalg.det(r), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("requires a rigid homogeneous transform")
    return t


def tool_in_board_from_fixed_camera(T_base_tool, T_base_camera, T_camera_board):
    """Use a camera rigidly bound to this arm base at the current body pose.

    Head/chest extrinsics for another arm base or body pose are not valid inputs.
    Both base translation AND rotation are compensated by matrix composition.
    """
    g, e, c = map(_rigid, (T_base_tool, T_base_camera, T_camera_board))
    return np.linalg.inv(c) @ np.linalg.inv(e) @ g


def tool_in_board_from_wrist_camera(T_tool_camera, T_camera_board):
    """Direct optical tool pose: does not require the controller base pose.

    Inherits wrist hand-eye error; it is not an independent hand-eye validation.
    The full, uniquely oriented board must be visible at each contact pose.
    """
    x, c = map(_rigid, (T_tool_camera, T_camera_board))
    return np.linalg.inv(c) @ np.linalg.inv(x)


def fixed_camera_candidate(
    T_base_tool, T_tool_wrist_camera, T_wrist_camera_board, T_fixed_camera_board
):
    """One simultaneous shared-board estimate; NOT a calibrated extrinsic.

    Base and board may move BETWEEN paired sets, but not within an unsynchronized
    pair. This avoids assuming one constant base-to-board pose across all sets.
    Multiple sets and held-out verification must qualify the resulting extrinsic.
    """
    g, x, w, c = map(
        _rigid,
        (T_base_tool, T_tool_wrist_camera, T_wrist_camera_board, T_fixed_camera_board),
    )
    return g @ x @ w @ np.linalg.inv(c)


def relative_base_motion(T_base_camera, T_camera_board_reference, T_camera_board_now):
    """Return T_base_reference_base_now for an unchanged base-camera transform."""
    e, c0, ci = map(
        _rigid, (T_base_camera, T_camera_board_reference, T_camera_board_now)
    )
    return (e @ c0) @ np.linalg.inv(e @ ci)


def fit_board_pivot(tool_poses_in_board):
    """Fit only supplied TRAINING poses; retain existing observability gates."""
    poses = [_rigid(p) for p in tool_poses_in_board]
    result = fit_pivot(poses)
    result["fixed_point_in_board_m"] = result.pop("fixed_point_in_base_m")
    result["pose_parent_frame"] = "board"
    result["qualified"] = False
    result["physical_compensation_verified"] = False
    return result


def evaluate_board_pivot(tool_poses_in_board, frozen_result):
    """Evaluate against the frozen point without fitting a new validation centre."""
    if frozen_result.get("pose_parent_frame") != "board":
        raise ValueError("result must use the board frame")
    raw = dict(frozen_result)
    raw["fixed_point_in_base_m"] = frozen_result["fixed_point_in_board_m"]
    metrics = evaluate_pivot([_rigid(p) for p in tool_poses_in_board], raw)
    metrics["refitted"] = False
    return metrics
