"""Opt-in simulation grasp monitor anchored at a fixed point on the handle.

The limits are experiment settings, not calibrated hardware force limits. This
module does not change the default rollout/evaluator or command any hardware.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class HandleGraspLimits:
    handle_translation_m: float = 0.005
    relative_rotation_deg: float = 5.0
    minimum_pad_force_n: float = 0.02
    unilateral_timeout_s: float = 0.050
    bilateral_timeout_s: float = 0.010
    maximum_sample_gap_s: float = 0.002

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError("limits must be positive and finite")


def _pose(value):
    value = np.asarray(value, dtype=float)
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1])
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(value[:3, :3]), 1.0, atol=1e-6)
    ):
        raise ValueError("need a finite rigid pose")
    return value


class HandleGraspMonitor:
    """Observe force continuity and handle motion without moving the anchor.

    ``relative_pose`` maps pot coordinates into the grasp frame. The fixed
    ``handle_local_m`` point avoids interpreting rotation of the remote pot
    origin as equivalent translation at the grasp. Alternating unloaded pads
    do not reset the unilateral timer. A missed sampling interval is a failure.
    """

    def __init__(self, reference_relative_pose, handle_local_m, limits=None):
        self.reference = _pose(reference_relative_pose).copy()
        self.handle = np.asarray(handle_local_m, dtype=float)
        if self.handle.shape != (3,) or not np.isfinite(self.handle).all():
            raise ValueError("need finite handle coordinates")
        self.handle = self.handle.copy()
        self.limits = limits or HandleGraspLimits()
        self.reference_point = self.reference[:3, :3] @ self.handle + self.reference[:3, 3]
        self.last_time = None
        self.unilateral_since = None
        self.bilateral_since = None

    def update(self, time_s, relative_pose, pad_forces_n):
        relative = _pose(relative_pose)
        forces = np.asarray(pad_forces_n, dtype=float)
        if (
            not np.isfinite(time_s)
            or forces.shape != (2,)
            or not np.isfinite(forces).all()
            or np.any(forces < 0)
            or (self.last_time is not None and time_s <= self.last_time)
        ):
            raise ValueError("need increasing finite time and two nonnegative forces")
        failures = []
        if (
            self.last_time is not None
            and time_s - self.last_time > self.limits.maximum_sample_gap_s + 1e-9
        ):
            failures.append("observation_gap")
        self.last_time = float(time_s)
        loaded = forces >= self.limits.minimum_pad_force_n
        for attr, invalid in [
            ("unilateral_since", not loaded.all()),
            ("bilateral_since", not loaded.any()),
        ]:
            since = getattr(self, attr)
            setattr(self, attr, (time_s if since is None else since) if invalid else None)
        unilateral = 0.0 if self.unilateral_since is None else time_s - self.unilateral_since
        bilateral = 0.0 if self.bilateral_since is None else time_s - self.bilateral_since
        delta = relative[:3, :3] @ self.handle + relative[:3, 3] - self.reference_point
        translation = float(np.linalg.norm(delta))
        angle = float(
            np.rad2deg(
                Rotation.from_matrix(relative[:3, :3] @ self.reference[:3, :3].T).magnitude()
            )
        )
        if translation > self.limits.handle_translation_m:
            failures.append("handle_translation")
        if angle > self.limits.relative_rotation_deg:
            failures.append("relative_rotation")
        if unilateral >= self.limits.unilateral_timeout_s - 1e-9:
            failures.append("persistent_unilateral_unload")
        if bilateral >= self.limits.bilateral_timeout_s - 1e-9:
            failures.append("persistent_bilateral_unload")
        return {
            "handle_delta_grasp_m": delta.tolist(),
            "handle_translation_m": translation,
            "relative_rotation_deg": angle,
            "pot_origin_translation_m": float(
                np.linalg.norm(relative[:3, 3] - self.reference[:3, 3])
            ),
            "pad_forces_n": forces.tolist(),
            "both_pads_loaded": bool(loaded.all()),
            "unilateral_unload_s": float(unilateral),
            "bilateral_unload_s": float(bilateral),
            "failures": failures,
        }
