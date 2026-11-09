import numpy as np
from scipy.spatial.transform import Rotation

from cap_harness.realman.held_object import stationary_grasp_target


def test_station_keeping_opposes_sag_without_reanchoring_grasp():
    target = np.eye(4)
    target[:3, 3] = [0.9, 0.4, 0.8]
    actual = target.copy()
    actual[2, 3] -= 0.002
    anchor = np.eye(4)
    anchor[0, 3] = 0.1
    originals = [x.copy() for x in [actual, target, anchor]]
    grasp, pos, angle = stationary_grasp_target(
        actual,
        target,
        anchor,
        np.zeros(3),
        np.zeros(3),
        [0, 0, -0.001],
        angular_limit_deg=5.5,
    )
    assert grasp[2, 3] > target[2, 3] and pos[2] > 0
    assert np.allclose(grasp[0, 3], 0.8)
    for source, original in zip([actual, target, anchor], originals):
        assert np.array_equal(source, original)


def test_feedback_authority_is_bounded_under_persistent_error():
    target, actual = np.eye(4), np.eye(4)
    actual[:3, 3] = [-1, 0, -1]
    actual[:3, :3] = Rotation.from_euler("x", 20, degrees=True).as_matrix()
    position, angular = np.zeros(3), np.zeros(3)
    for _ in range(1000):
        _, position, angular = stationary_grasp_target(
            actual,
            target,
            np.eye(4),
            position,
            angular,
            np.zeros(3),
            angular_limit_deg=5.5,
        )
    assert np.linalg.norm(position) <= 0.0200000001
    assert np.linalg.norm(angular) <= np.deg2rad(5.500000001)
