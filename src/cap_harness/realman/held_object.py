"""Station keeping for a grasped simulated object while the other arm works.

Uses the same 20 ms integral/PD correction as the coffee object-motion controller.
It does not change the object target, grasp anchor, actuator limits or guards.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .tracking import bounded_rotation_integral


def stationary_grasp_target(
    actual,
    target,
    grasp_to_object,
    position_integral,
    angular_integral,
    linear_velocity,
    *,
    angular_limit_deg,
    loaded_pd=True,
):
    transforms = [np.asarray(x, dtype=float) for x in [actual, target, grasp_to_object]]
    vectors = [
        np.asarray(x, dtype=float) for x in [position_integral, angular_integral, linear_velocity]
    ]
    if any(x.shape != (4, 4) or not np.isfinite(x).all() for x in transforms):
        raise ValueError("finite object/grasp transforms required")
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in vectors):
        raise ValueError("finite feedback and velocity vectors required")
    actual, target, grasp_to_object = transforms
    position_integral, angular_integral, linear_velocity = vectors
    error = target[:3, 3] - actual[:3, 3]
    position = position_integral + (0.01 if loaded_pd else 0.04) * error
    position *= min(1.0, 0.02 / max(float(np.linalg.norm(position)), 1e-12))
    angular = bounded_rotation_integral(
        angular_integral,
        Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec(),
        maximum_deg=angular_limit_deg,
    )
    grasp = target @ np.linalg.inv(grasp_to_object)
    grasp[:3, 3] += position
    grasp[:3, :3] = Rotation.from_rotvec(angular).as_matrix() @ grasp[:3, :3]
    if loaded_pd:
        grasp[:3, 3] += 0.8 * error - 0.1 * linear_velocity
    return grasp, position, angular
