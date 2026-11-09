"""Glare rejection must retain physical-motion and independent-validation failures."""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cap_harness.realman.handeye_solver import project, transform
from cap_harness.realman.shared_board_calibration import (
    board_pose_upright, stationary_corner_check, fit_fixed_camera, evaluate_fixed_camera,
)


def detection(monkeypatch, shift=(0, 0), bad_ids=(18,)):
    board = dict(type="chessboard", inner_corners_columns=11, inner_corners_rows=8, square_size_m=0.06)
    objects = np.zeros((88, 3))
    objects[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * 0.06
    k = np.array([[610., 0, 640], [0, 610, 360], [0, 0, 1]])
    info = dict(k=k.ravel().tolist(), d=[0.] * 5, distortion_model="plumb_bob", width=1280, height=720)
    points = cv2.projectPoints(objects, np.array([0.2, -0.15, 0.03]), np.array([-0.3, -0.21, 0.9]), k, np.zeros(5))[0].reshape(-1, 2)
    points += shift
    points[list(bad_ids)] += [13, 0]
    original = points.copy()
    monkeypatch.setattr(cv2, "findChessboardCornersSB", lambda *a, **kw: (True, points.reshape(-1, 1, 2).copy()))
    result = board_pose_upright(np.zeros((720, 1280), np.uint8), info, board, robust_corners=True)
    np.testing.assert_array_equal(result["all_corners"], original)
    return result


def test_reflection_outliers_do_not_change_measured_pixels_or_stationary_truth(monkeypatch):
    a = detection(monkeypatch, bad_ids=(18,))
    b = detection(monkeypatch, bad_ids=(27,))
    report = stationary_corner_check(a, b)
    assert report["excluded_union_ids"] == [18, 27]
    assert report["common_count"] == 86
    assert report["raw_motion_max_px"] > 12
    assert report["motion_max_px"] < 1e-8
    np.testing.assert_allclose(a["C"], b["C"], atol=1e-7)


def test_real_board_motion_still_fails_with_glare(monkeypatch):
    a = detection(monkeypatch)
    b = detection(monkeypatch, shift=(2, 0))
    with pytest.raises(ValueError, match="changed within set"):
        stationary_corner_check(a, b)


def test_two_individual_good_masks_cannot_remove_too_many_common_points(monkeypatch):
    a = detection(monkeypatch, bad_ids=(18, 29, 40))
    b = detection(monkeypatch, bad_ids=(16, 27, 38))
    with pytest.raises(ValueError, match="too few common"):
        stationary_corner_check(a, b)


def test_variable_subsets_preserve_fixed_camera_fit_and_bad_holdout_detection():
    rng = np.random.default_rng(24)
    e = transform(Rotation.from_euler("xyz", [.1, -.2, .15]).as_matrix(), [.08, -.1, .2])
    objects = np.zeros((88, 3))
    objects[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * .06
    k = np.array([[610., 0, 640], [0, 610, 360], [0, 0, 1]])
    d = np.zeros(5)
    records = []
    for i in range(9):
        c = transform(Rotation.from_rotvec(rng.uniform(-.2, .2, 3)).as_matrix(), rng.uniform(-.1, .1, 3) + [0, 0, 1.1])
        kept = np.delete(objects, [i + 12, i + 30], axis=0)
        records.append(dict(sample=str(i), Y=e @ c, C=c, objects=kept, corners=project(kept, c, k, d)))
    actual = fit_fixed_camera(records[:6], objects, k, d)
    np.testing.assert_allclose(actual, e, atol=1e-8)
    good = evaluate_fixed_camera(records[6:], objects, k, d, actual)
    assert max(v["board_center_mm"] for v in good) < 1e-6
    records[6]["Y"][0, 3] += .012
    bad = evaluate_fixed_camera(records[6:], objects, k, d, actual)
    assert bad[0]["translation_mm"] > 11
    assert bad[0]["reprojection_rms_px"] > 5
