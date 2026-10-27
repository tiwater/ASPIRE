import json

import numpy as np
import pytest

from cap_harness.realman.calibration import board_spec, chessboard_pose_candidates, inspect_capture


def test_unknown_board_cannot_get_invented_dimensions():
    assert board_spec({"type": None}) is None
    with pytest.raises(ValueError):
        board_spec(
            {
                "type": "chessboard",
                "inner_corners_columns": 9,
                "inner_corners_rows": 6,
                "square_size_m": None,
            }
        )


def test_pose_candidate_recovers_synthetic_pose_without_claiming_calibration():
    cv2 = pytest.importorskip("cv2")
    spec = {
        "type": "chessboard",
        "inner_corners_columns": 11,
        "inner_corners_rows": 8,
        "square_size_m": 0.06,
    }
    k = np.array([[700.0, 0, 640], [0, 710, 360], [0, 0, 1]])
    points = np.zeros((88, 3))
    points[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * 0.06
    rvec, tvec = np.array([0.2, -0.15, 0.03]), np.array([-0.3, -0.21, 1.2])
    pixels = cv2.projectPoints(points, rvec, tvec, k, np.zeros(5))[0].reshape(-1, 2)
    info = {
        "k": k.ravel().tolist(),
        "d": [0] * 5,
        "distortion_model": "plumb_bob",
        "width": 1280,
        "height": 720,
    }
    result = chessboard_pose_candidates(pixels, info, spec)
    assert len(result["candidates"]) == 2
    best = result["candidates"][0]
    np.testing.assert_allclose(np.array(best["T_camera_board"])[:3, 3], tvec, atol=1e-6)
    assert best["reprojection_rms_px"] < 1e-6
    assert not result["metric_accuracy_verified"] and not result["handeye_qualified"]
    info["distortion_model"] = "equidistant"
    with pytest.raises(ValueError):
        chessboard_pose_candidates(pixels, info, spec)


def test_detect_board_does_not_qualify_handeye(tmp_path):
    pytest.importorskip("cv2")
    gray = np.full((360, 480), 255, dtype=np.uint8)
    for y in range(7):
        for x in range(10):
            if (x + y) % 2 == 0:
                gray[40 + y * 40 : 80 + y * 40, 40 + x * 40 : 80 + x * 40] = 0
    rgb = np.repeat(gray[..., None], 3, axis=2)
    (tmp_path / "frame.bin").write_bytes(rgb.tobytes())
    meta = {
        "encoding": "rgb8",
        "is_bigendian": False,
        "width": 480,
        "height": 360,
        "step": 1440,
        "file": "frame.bin",
        "header": {"frame_id": "optical"},
    }
    info = {
        "width": 480,
        "height": 360,
        "k": [500, 0, 240, 0, 500, 180, 0, 0, 1],
        "d": [0] * 5,
        "distortion_model": "plumb_bob",
        "header": {"frame_id": "optical"},
    }
    data = {
        "started_unix_s": 1,
        "images": {"/test/color": meta},
        "camera_info": {"/test/color": {"message": info}},
    }
    (tmp_path / "capture.json").write_text(json.dumps(data))
    spec = {
        "type": "chessboard",
        "inner_corners_columns": 9,
        "inner_corners_rows": 6,
        "square_size_m": 0.02,
    }
    report = inspect_capture(tmp_path, spec)
    camera = report["cameras"]["/test/color"]
    assert camera["board_detected"] and len(camera["corners_px"]) == 54
    assert camera["intrinsics_metadata_consistent"]
    assert not report["handeye_qualified"] and not report["controller_pose_pairs_available"]
    data["camera_info"]["/test/color"]["message"]["header"]["frame_id"] = "wrong"
    (tmp_path / "capture.json").write_text(json.dumps(data))
    assert not inspect_capture(tmp_path, spec)["cameras"]["/test/color"][
        "intrinsics_metadata_consistent"
    ]
