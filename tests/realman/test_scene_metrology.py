"""Offline measurement checks with known metric geometry and bad data."""

import numpy as np
import pytest

from cap_harness.realman.scene_metrology import (
    fit_circle_pixels,
    fit_plane,
    fit_radial_surface,
    intersect_pixels,
    measure_segment,
)


def test_plane_recovers_tilted_table_despite_foreground_outliers():
    rng = np.random.default_rng(3)
    x, z = rng.uniform(-0.2, 0.2, 1000), rng.uniform(0.4, 0.8, 1000)
    y = 0.3 - 0.2 * z + rng.normal(0, 0.0004, 1000)
    points = np.column_stack([x, y, z])
    points[:150, 1] -= 0.05
    result = fit_plane(points)
    expected = np.array([0, -1, -0.2]) / np.sqrt(1.04)
    assert np.dot(result["normal"], expected) > 0.99999
    assert result["offset"] == pytest.approx(0.3 / np.sqrt(1.04), abs=0.0001)
    assert 0.83 < result["inlier_fraction"] < 0.87


def test_metric_segment_and_pixel_annotation_envelope():
    info = {"distortion_model": "plumb_bob", "k": [600, 0, 640, 0, 600, 360, 0, 0, 1], "d": [0] * 5}
    plane = {"normal": [0, 0, 1], "offset": -0.5}
    result = measure_segment([[580, 360], [700, 360]], info, plane)
    assert result["length_m"] == pytest.approx(0.1)
    assert result["annotation_only_range_m"][0] < 0.1 < result["annotation_only_range_m"][1]
    assert result["metric_accuracy_qualified"] is False
    with pytest.raises(ValueError, match="grazing"):
        intersect_pixels([[640, 360]], info, {"normal": [1, 0, 0], "offset": -0.1})


def test_partial_cup_surface_recovers_known_frustum():
    rng = np.random.default_rng(5)
    angle = rng.uniform(-1.1, 1.1, 900)
    height = rng.uniform(0.01, 0.075, 900)
    radius = 0.025 + 0.125 * height
    points = np.column_stack([radius * np.sin(angle), 0.25 - height, 0.55 - radius * np.cos(angle)])
    points += rng.normal(0, 0.00015, points.shape)
    result = fit_radial_surface(points, {"normal": [0, -1, 0], "offset": 0.25}, tapered=True)
    assert result["radius_at_table_height_m"] == pytest.approx(0.025, abs=0.0004)
    assert result["slope"] == pytest.approx(0.125, abs=0.005)


def test_unknown_or_degenerate_depth_cannot_become_a_plane():
    with pytest.raises(ValueError):
        fit_plane(np.full((100, 3), np.nan))
    with pytest.raises(ValueError):
        fit_plane(np.column_stack([np.linspace(0, 1, 100), np.zeros((100, 2))]))


def test_circle_diameter_recovers_metric_size_from_perspective_boundary():
    import cv2

    info = {"distortion_model": "plumb_bob", "k": [600, 0, 640, 0, 600, 360, 0, 0, 1], "d": [0] * 5}
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    normal = np.array([0, -0.8, -0.6])
    center = np.array([0.1, 0.15, 0.7])
    points = center + 0.06 * (
        np.cos(angles)[:, None] * [1, 0, 0] + np.sin(angles)[:, None] * [0, -0.6, 0.8]
    )
    pixels = cv2.projectPoints(
        points, np.zeros(3), np.zeros(3), np.array(info["k"], float).reshape(3, 3), np.zeros(5)
    )[0][:, 0]
    result = fit_circle_pixels(pixels, info, {"normal": normal, "offset": -normal @ center})
    assert result["diameter_m"] == pytest.approx(0.12, abs=1e-6)
