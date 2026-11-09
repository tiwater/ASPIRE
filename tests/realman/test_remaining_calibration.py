import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cap_harness.realman.handeye_solver import project, transform
from cap_harness.realman.shared_board_calibration import evaluate_fixed_camera, fit_fixed_camera
from cap_harness.realman.tcp_calibration import evaluate_pivot, fit_pivot


def test_static_camera_direction_and_independent_board_validation():
    rng = np.random.default_rng(5)
    e = transform(Rotation.from_euler("xyz", [0.1, -0.2, 0.15]).as_matrix(), [0.08, -0.1, 0.2])
    obj = np.zeros((24, 3))
    obj[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2) * 0.04
    k = np.array([[610.0, 0, 640], [0, 610, 360], [0, 0, 1]])
    d = np.zeros(5)
    records = []
    for i in range(9):
        c = transform(
            Rotation.from_rotvec(rng.uniform(-0.2, 0.2, 3)).as_matrix(),
            rng.uniform(-0.1, 0.1, 3) + [0, 0, 0.8],
        )
        records.append({"sample": str(i), "Y": e @ c, "C": c, "corners": project(obj, c, k, d)})
    actual = fit_fixed_camera(records[:6], obj, k, d)
    np.testing.assert_allclose(actual, e, atol=1e-8)
    good = evaluate_fixed_camera(records[6:], obj, k, d, actual)
    assert max(r["translation_mm"] for r in good) < 1e-6
    records[6]["Y"][0, 3] += 0.012
    bad = evaluate_fixed_camera(records[6:], obj, k, d, actual)
    assert bad[0]["translation_mm"] > 11
    assert bad[0]["reprojection_rms_px"] > 5


def test_pivot_recovers_probe_but_does_not_claim_grasp_center_or_orientation():
    rng = np.random.default_rng(7)
    tip, fixed = np.array([0.01, -0.02, 0.2]), np.array([0.3, 0.1, 0.4])
    poses = []
    for _ in range(10):
        r = Rotation.from_rotvec(rng.uniform(-0.4, 0.4, 3)).as_matrix()
        poses.append(transform(r, fixed - r @ tip))
    result = fit_pivot(poses[:7])
    np.testing.assert_allclose(result["probe_tip_in_tool_m"], tip, atol=1e-10)
    assert evaluate_pivot(poses[7:], result)["max_mm"] < 1e-6
    assert not result["grasp_center_calibrated"]
    assert not result["tool_orientation_calibrated"]
    poses[7][2, 3] += 0.008
    assert evaluate_pivot(poses[7:], result)["max_mm"] > 7


def test_pivot_rejects_same_axis():
    poses = [
        transform(Rotation.from_euler("z", t).as_matrix(), [0, 0, 0])
        for t in np.linspace(0, 0.5, 8)
    ]
    with pytest.raises(ValueError, match="diverse"):
        fit_pivot(poses)
