import numpy as np
import pytest

from cap_harness.realman.grasp_capacity import minimum_pad_load


def contacts(mu=0.5):
    return [
        {
            "pad": "left",
            "position_m": [-0.03, 0, 0],
            "frame": np.eye(3),
            "sliding_friction": mu,
            "torsional_friction_m": 0,
        },
        {
            "pad": "right",
            "position_m": [0.03, 0, 0],
            "frame": np.diag([-1, -1, 1]),
            "sliding_friction": mu,
            "torsional_friction_m": 0,
        },
    ]


def test_symmetric_friction_grasp_matches_analytic_load():
    result = minimum_pad_load(contacts(), [0, 0, -10, 0, 0, 0], [0, 0, 0])
    assert result["static_model_feasible"]
    assert result["minimum_max_pad_normal_n"] == pytest.approx(10)
    assert np.max(np.abs(result["equilibrium_residual"])) < 1e-8
    assert not result["hardware_qualified"]


def test_zero_friction_cannot_support_vertical_load():
    assert not minimum_pad_load(contacts(0), [0, 0, -10, 0, 0, 0], [0, 0, 0])[
        "static_model_feasible"
    ]


def test_missing_moment_capacity_is_not_treated_as_force_success():
    result = minimum_pad_load(contacts(), [0, 0, -10, 1, 0, 0], [0, 0, 0])
    assert not result["static_model_feasible"]


def test_invalid_frame_rejected():
    points = contacts()
    points[0]["frame"] = np.zeros((3, 3))
    with pytest.raises(ValueError):
        minimum_pad_load(points, [0, 0, -10, 0, 0, 0], [0, 0, 0])


def test_relaxed_force_bound_is_below_elliptic_requirement():
    points = contacts()
    for point in points:
        point["torsional_friction_m"] = 0.005
    wrench = [0, 0, -10, 0.1, 0, 0]
    inner = minimum_pad_load(points, wrench, [0, 0, 0])
    outer = minimum_pad_load(points, wrench, [0, 0, 0], friction_model="outer_box")
    assert outer["minimum_max_pad_normal_n"] == pytest.approx(10)
    # Sliding and torsional friction share the elliptic cone; they cannot each
    # independently consume its full capacity at the same normal force.
    assert inner["minimum_max_pad_normal_n"] >= np.sqrt(10**2 + 10**2) - 1e-8
    assert outer["force_bound"] == "optimistic_lower"


def test_world_frame_rotation_and_translation_preserve_required_force():
    from scipy.spatial.transform import Rotation

    points = contacts()
    rotation = Rotation.from_euler("xyz", [0.3, 0.8, -0.4]).as_matrix()
    shift = np.array([0.4, -0.6, 1.3])
    for point in points:
        point["position_m"] = rotation @ point["position_m"] + shift
        point["frame"] = point["frame"] @ rotation.T
    force = rotation @ np.array([0, 0, -10])
    result = minimum_pad_load(points, np.r_[force, [0, 0, 0]], shift)
    assert result["minimum_max_pad_normal_n"] == pytest.approx(10)
