import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cap_harness.realman.handeye_solver import transform
from cap_harness.realman.moving_base_tcp import (
    evaluate_board_pivot,
    fit_board_pivot,
    fixed_camera_candidate,
    relative_base_motion,
    tool_in_board_from_fixed_camera,
    tool_in_board_from_wrist_camera,
)
from cap_harness.realman.tcp_calibration import evaluate_pivot, fit_pivot


def moving_fixture():
    rng = np.random.default_rng(20261027)
    tip, point = np.array([0.01, -0.015, 0.41]), np.array([0.2, 0.1, 0.15])
    e = transform(Rotation.from_euler("xyz", [0.2, -0.1, 1.1]).as_matrix(), [0.1, 0.2, 0.3])
    x = transform(Rotation.from_euler("xyz", [-0.05, 0.1, 1.5]).as_matrix(), [-0.08, 0.02, 0.01])
    rows = []
    for _ in range(12):
        rotation = Rotation.from_rotvec(rng.uniform(-0.6, 0.6, 3)).as_matrix()
        h = transform(rotation, point - rotation @ tip)  # board_T_tool
        base = transform(
            Rotation.from_rotvec(rng.uniform(-0.04, 0.04, 3)).as_matrix(),
            rng.uniform(-0.025, 0.025, 3),
        )  # board_T_base (six-DoF drift)
        g = np.linalg.inv(base) @ h
        c, w = np.linalg.inv(base @ e), np.linalg.inv(h @ x)
        rows.append((g, c, w, h, base))
    return tip, point, e, x, rows


def test_compensation_recovers_tcp_with_six_dof_drift_and_frozen_holdout():
    tip, point, e, x, rows = moving_fixture()
    corrected = [tool_in_board_from_fixed_camera(g, e, c) for g, c, _, _, _ in rows]
    for actual, (_, _, _, expected, _) in zip(corrected, rows):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    result = fit_board_pivot(corrected[:8])
    np.testing.assert_allclose(result["probe_tip_in_tool_m"], tip, atol=1e-12)
    np.testing.assert_allclose(result["fixed_point_in_board_m"], point, atol=1e-12)
    before = copy.deepcopy(result)
    assert evaluate_board_pivot(corrected[8:], result)["max_mm"] < 1e-8
    bad = [p.copy() for p in corrected[8:]]
    bad[0][0, 3] += 0.008  # reference/pointing error must not be absorbed
    assert evaluate_board_pivot(bad, result)["max_mm"] > 7.9
    assert result == before
    assert not result["qualified"] and not result["grasp_center_calibrated"]
    raw = [row[0] for row in rows]
    assert evaluate_pivot(raw, fit_pivot(raw))["rms_mm"] > 5
    optical = [tool_in_board_from_wrist_camera(x, row[2]) for row in rows]
    np.testing.assert_allclose(optical, corrected, atol=1e-12)


def test_shared_board_extrinsic_and_relative_base_direction():
    _, _, e, x, rows = moving_fixture()
    for g, c, w, _, base in rows:
        np.testing.assert_allclose(fixed_camera_candidate(g, x, w, c), e, atol=1e-12)
        delta = relative_base_motion(e, rows[0][1], c)
        np.testing.assert_allclose(delta, np.linalg.inv(rows[0][4]) @ base, atol=1e-12)


def test_stale_tracking_is_not_valid_compensation():
    _, _, e, _, rows = moving_fixture()
    good = [tool_in_board_from_fixed_camera(g, e, c) for g, c, _, _, _ in rows]
    frozen = fit_board_pivot(good[:8])
    stale = [tool_in_board_from_fixed_camera(row[0], e, rows[0][1]) for row in rows[8:]]
    assert evaluate_board_pivot(stale, frozen)["rms_mm"] > 5


@pytest.mark.parametrize("kind", ["shape", "nan", "scale", "reflection", "last_row"])
def test_reject_non_rigid_or_malformed_transforms(kind):
    bad = np.eye(4)
    if kind == "shape":
        bad = np.eye(3)
    elif kind == "nan":
        bad[0, 3] = np.nan
    elif kind == "scale":
        bad[0, 0] = 2
    elif kind == "reflection":
        bad[0, 0] = -1
    else:
        bad[3, 0] = 0.1
    with pytest.raises(ValueError):
        tool_in_board_from_fixed_camera(bad, np.eye(4), np.eye(4))
