"""Known geometry, corrupt observations, and independent holdout checks."""

import cv2
import numpy as np
import pytest

from cap_harness.realman.calibration import chessboard_pose_with_bounded_rejection


def synthetic_board():
    board = dict(
        type="chessboard", inner_corners_columns=11, inner_corners_rows=8, square_size_m=0.06
    )
    objects = np.zeros((88, 3))
    objects[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * 0.06
    k = np.array([[610.0, 0, 640], [0, 610, 360], [0, 0, 1]])
    rv, tv = np.array([0.2, -0.15, 0.03]), np.array([-0.3, -0.21, 0.9])
    pixels = cv2.projectPoints(objects, rv, tv, k, np.zeros(5))[0].reshape(-1, 2)
    info = dict(
        k=k.ravel().tolist(), d=[0] * 5, width=1280, height=720, distortion_model="plumb_bob"
    )
    return board, info, pixels, rv, tv


def test_single_bad_corner_recovers_truth_without_changing_observations():
    board, info, pixels, rv, tv = synthetic_board()
    pixels[18] += [13, 0]
    original = pixels.copy()
    report = chessboard_pose_with_bounded_rejection(pixels, info, board)
    assert report["excluded_ids"] == [18]
    assert len(report["retained_ids"]) == 87
    assert report["retained_rms_px"] < 1e-6
    recovered = np.asarray(report["T_camera_board"])
    np.testing.assert_allclose(recovered[:3, 3], tv, atol=1e-7)
    np.testing.assert_allclose(recovered[:3, :3], cv2.Rodrigues(rv)[0], atol=1e-7)
    np.testing.assert_array_equal(pixels, original)
    assert not report["handeye_qualified"]


def test_clean_frame_retains_every_corner():
    board, info, pixels, *_ = synthetic_board()
    report = chessboard_pose_with_bounded_rejection(pixels, info, board)
    assert report["excluded_ids"] == []
    assert report["retained_rms_px"] < 1e-6


def test_multiple_corrupt_corners_reject_entire_frame():
    board, info, pixels, *_ = synthetic_board()
    pixels[[13, 24, 35, 46, 57]] += [0, 15]
    with pytest.raises(ValueError, match="too many outlier"):
        chessboard_pose_with_bounded_rejection(pixels, info, board)


def test_distributed_noise_cannot_pass_by_trimming_until_fit_is_good():
    board, info, pixels, *_ = synthetic_board()
    pixels += np.random.default_rng(29).normal(0, 2.5, pixels.shape)
    with pytest.raises(ValueError):
        chessboard_pose_with_bounded_rejection(pixels, info, board)
