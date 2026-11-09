"""A turn target follows an object-centred arc, not a wrist-centred twist."""

import numpy as np
import pytest

from cap_harness.realman.grasp_geometry import pivot_yaw_pose


def test_turn_keeps_rigidly_grasped_pot_centre_fixed_and_is_reversible():
    pot = np.eye(4)
    pot[:3, 3] = [0.8, 0.2, 0.796]
    relative = np.eye(4)
    relative[:3, 3] = [0.115, 0, 0.081]
    grasp = pot @ relative
    turned = pivot_yaw_pose(grasp, pot[:3, 3], np.pi / 2)
    np.testing.assert_allclose(turned[:3, 3], [0.8, 0.315, 0.877])
    inferred_pot = turned @ np.linalg.inv(relative)
    np.testing.assert_allclose(inferred_pot[:3, 3], pot[:3, 3])
    np.testing.assert_allclose(inferred_pot[:3, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(pivot_yaw_pose(turned, pot[:3, 3], -np.pi / 2), grasp)
    np.testing.assert_allclose(grasp, pot @ relative)  # input was not mutated


def test_turn_rejects_nonfinite_target():
    with pytest.raises(ValueError, match="finite"):
        pivot_yaw_pose(np.eye(4), [0, 0, 0], np.nan)
