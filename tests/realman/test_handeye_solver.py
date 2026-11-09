"""Synthetic ground truth and independent bad-pose detection; no robot access."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cap_harness.realman.handeye_solver import (
    controller_transform,
    evaluate,
    pose_difference,
    project,
    refine_pixels,
    solve_closed_form,
    transform,
)


def scene():
    rng = np.random.default_rng(17)
    x = transform(Rotation.from_euler("xyz", [0.1, -0.12, 1.5]).as_matrix(), [0.08, -0.02, 0.015])
    y = transform(Rotation.from_euler("xyz", [0.15, 0.25, -0.1]).as_matrix(), [-0.1, -0.05, 0.8])
    gs = [
        transform(
            Rotation.from_rotvec(rng.uniform(-0.15, 0.15, 3)).as_matrix(),
            rng.uniform(-0.03, 0.03, 3),
        )
        for _ in range(16)
    ]
    cs = [np.linalg.inv(g @ x) @ y for g in gs]
    obj = np.zeros((24, 3))
    obj[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2) * 0.04
    k = np.array([[610.0, 0, 640], [0, 610, 360], [0, 0, 1]])
    d = np.zeros(5)
    corners = [project(obj, c, k, d) for c in cs]
    return x, y, gs, cs, obj, k, d, corners


def test_ground_truth_direction_and_metric_units():
    x, y, gs, cs, *_ = scene()
    xx, yy = solve_closed_form(gs[:12], cs[:12])
    assert pose_difference(xx, x)["translation_mm"] < 1e-6
    assert pose_difference(xx, x)["rotation_deg"] < 1e-6
    assert pose_difference(yy, y)["translation_mm"] < 1e-6
    for g, c in zip(gs[12:], cs[12:]):
        np.testing.assert_allclose(g @ xx @ c, yy, atol=1e-9)


def test_pixel_fit_does_not_absorb_changed_holdout_pose():
    x, y, gs, cs, obj, k, d, corners = scene()
    xx, yy = solve_closed_form(gs[:12], cs[:12])
    xx, yy, result = refine_pixels(gs[:12], corners[:12], obj, k, d, xx, yy)
    assert result["rms_corner_px"] < 1e-6
    records = [
        {"sample": str(i), "G": gs[i].copy(), "C": cs[i], "corners": corners[i]}
        for i in range(12, 16)
    ]
    assert evaluate(records, obj, k, d, xx, yy)["summary"]["translation_mm_max"] < 1e-6
    records[0]["G"][0, 3] += 0.015
    bad = evaluate(records, obj, k, d, xx, yy)["summary"]
    assert bad["translation_mm_max"] > 14
    assert bad["reprojection_rms_px_max"] > 5
    np.testing.assert_allclose(xx, x, atol=1e-9)


def test_rejects_one_axis_or_tiny_excitation():
    gs = [
        transform(Rotation.from_euler("z", t).as_matrix(), [0, 0, 0])
        for t in np.linspace(0, 0.4, 12)
    ]
    with pytest.raises(ValueError, match="excitation"):
        solve_closed_form(gs, [np.linalg.inv(g) for g in gs])


def test_variable_corner_masks_preserve_metric_fit_and_bad_holdout_detection():
    x, y, gs, cs, obj, k, d, corners = scene()
    masks = [np.arange(len(obj)) != i % len(obj) for i in range(len(gs))]
    view_objects = [obj[mask] for mask in masks]
    measured = [c[mask] for c, mask in zip(corners, masks, strict=True)]
    xx, yy = solve_closed_form(gs[:12], cs[:12])
    xx, yy, result = refine_pixels(
        gs[:12],
        measured[:12],
        obj,
        k,
        d,
        xx,
        yy,
        objects_per_view=view_objects[:12],
    )
    assert result["rms_corner_px"] < 1e-6
    np.testing.assert_allclose(xx, x, atol=1e-8)
    np.testing.assert_allclose(yy, y, atol=1e-8)
    records = [
        dict(
            sample=str(i),
            G=gs[i].copy(),
            C=cs[i],
            corners=measured[i],
            object_points=view_objects[i],
        )
        for i in range(12, 16)
    ]
    assert evaluate(records, obj, k, d, xx, yy)["summary"]["board_center_mm_max"] < 1e-6
    records[0]["G"][0, 3] += 0.015
    bad = evaluate(records, obj, k, d, xx, yy)["summary"]
    assert bad["translation_mm_max"] > 14
    assert bad["reprojection_rms_px_max"] > 5


def test_tsai_cannot_return_silent_identity_for_small_rotations():
    x, y, _, _, *_ = scene()
    gs = [
        transform(Rotation.from_euler("xy", [a, b], degrees=True).as_matrix(), [0, 0, 0])
        for a in [-3, 0, 3]
        for b in [-4, 0, 4]
    ]
    cs = [np.linalg.inv(g @ x) @ y for g in gs]
    with pytest.raises(ValueError, match="Tsai unsupported"):
        solve_closed_form(gs, cs, "TSAI")
    xx, _ = solve_closed_form(gs, cs, "PARK")
    np.testing.assert_allclose(xx, x, atol=1e-8)


def test_controller_fixed_xyz_convention():
    raw = [-363027, 189917, -326646, -2707, -1346, 771]
    t = controller_transform(raw)
    rx, ry, rz = [
        Rotation.from_euler(axis, angle / 1000).as_matrix() for axis, angle in zip("xyz", raw[3:])
    ]
    np.testing.assert_allclose(t[:3, :3], rz @ ry @ rx)
    np.testing.assert_allclose(t[:3, 3], [-0.363027, 0.189917, -0.326646])
