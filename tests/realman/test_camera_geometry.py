import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cap_harness.realman.camera_geometry import (
    camera_points_in_controller_base,
    load_wrist_calibration,
)


def test_current_right_profile_loads_without_upgrading_qualification():
    path = Path("configs/realman/calibration/right_wrist_20261028.json")
    before = path.read_bytes()
    profile = json.loads(before)
    info = {**profile["camera_info"], "header": {"frame_id": profile["child_frame"]}}
    transform = load_wrist_calibration(
        path, arm="right", camera_info=info,
        tool_frame={"tool_name": "Arm_Tip", "pose": [0] * 6},
        work_frame={"frame_name": "World", "pose": [0] * 6},
    )
    np.testing.assert_equal(transform, profile["T_tool_camera"])
    assert path.read_bytes() == before
    assert profile["general_workspace_qualified"] is False
    assert profile["tcp_qualified"] is False


def test_optical_points_use_forward_camera_to_tool_to_base():
    x = np.eye(4)
    x[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    x[:3, 3] = [0.08, -0.02, 0.01]
    points = np.array([[1.0, 0, 0], [0, 0, 1]])
    result = camera_points_in_controller_base(points, [100000, 200000, 300000, 0, 0, 0], x)
    np.testing.assert_allclose(result, [[0.18, 1.18, 0.31], [0.18, 0.18, 1.31]], atol=1e-12)


def test_loading_rejects_wrong_arm_mode_or_changed_tcp(tmp_path):
    path = tmp_path / "calibration.json"
    info = {
        "header": {"frame_id": "left_optical"},
        "width": 1280,
        "height": 720,
        "k": [610.0, 0, 640, 0, 610, 360, 0, 0, 1],
        "d": [0.0] * 5,
        "distortion_model": "plumb_bob",
    }
    path.write_text(
        json.dumps(
            {
                "status": "validated_local_geometry",
                "arm": "left",
                "parent_frame": "left_controller_Arm_Tip",
                "child_frame": "left_optical",
                "camera_info": info,
                "T_tool_camera": np.eye(4).tolist(),
            }
        )
    )
    kw = {
        "arm": "left",
        "camera_info": info,
        "tool_frame": {"tool_name": "Arm_Tip", "pose": [0] * 6},
        "work_frame": {"frame_name": "World", "pose": [0] * 6},
    }
    np.testing.assert_equal(load_wrist_calibration(path, **kw), np.eye(4))
    with pytest.raises(ValueError, match="wrong arm"):
        load_wrist_calibration(path, **{**kw, "arm": "right"})
    info["width"] = 640
    with pytest.raises(ValueError, match="intrinsics"):
        load_wrist_calibration(path, **kw)
    info["width"] = 1280
    kw["tool_frame"]["pose"][2] = 1000
    with pytest.raises(ValueError, match="offset"):
        load_wrist_calibration(path, **kw)
