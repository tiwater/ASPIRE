"""Use a reviewed wrist calibration for geometry only; never commands the robot."""

import json
from pathlib import Path

import numpy as np

from .handeye_solver import controller_transform, transform


def load_wrist_calibration(path, *, arm, camera_info, tool_frame, work_frame):
    """Load a locally validated geometric transform; never authorize workspace motion.

    The board-validation label retains its original local scope. Callers still
    need to assess that scope and contemporaneous pose evidence for their task.
    """
    calibration = json.loads(Path(path).read_text())
    if calibration.get("status") not in ("validated_local_geometry", "validated_local_board_geometry") or calibration.get("arm") != arm:
        raise ValueError("wrong arm or calibration has not passed local validation")
    if calibration.get("parent_frame") != arm + "_controller_Arm_Tip":
        raise ValueError("unexpected calibration parent frame")
    if tool_frame.get("tool_name") != "Arm_Tip" or work_frame.get("frame_name") != "World":
        raise ValueError("tool/work frame changed")
    if tool_frame.get("pose") != [0] * 6 or work_frame.get("pose") != [0] * 6:
        raise ValueError("tool/work offset changed")
    reference = calibration["camera_info"]
    if camera_info["header"]["frame_id"] != calibration["child_frame"]:
        raise ValueError("wrong camera optical frame")
    for key in ["width", "height", "k", "d", "distortion_model"]:
        if camera_info[key] != reference[key]:
            raise ValueError("camera mode or intrinsics changed")
    x = np.asarray(calibration["T_tool_camera"], dtype=float)
    if x.shape != (4, 4) or not np.allclose(x[3], [0, 0, 0, 1], atol=1e-12):
        raise ValueError("invalid homogeneous transform")
    transform(x[:3, :3], x[:3, 3])
    return x


def camera_points_in_controller_base(points_camera_m, controller_pose_raw, T_tool_camera):
    """Transform metric optical-frame points using a contemporaneous measured pose.

    Caller must supply synchronized/static pose and image evidence and use the
    matching arm. Returns the controller arm-base coordinates, NOT URDF/world/TCP.
    """
    points = np.asarray(points_camera_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("requires finite Nx3 points in metres")
    x = np.asarray(T_tool_camera, dtype=float)
    if x.shape != (4, 4) or not np.allclose(x[3], [0, 0, 0, 1], atol=1e-12):
        raise ValueError("invalid homogeneous transform")
    transform(x[:3, :3], x[:3, 3])
    base_T_camera = controller_transform(controller_pose_raw) @ x
    return points @ base_T_camera[:3, :3].T + base_T_camera[:3, 3]
