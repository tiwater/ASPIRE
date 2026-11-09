"""Conservative broad-phase for offline MuJoCo distance sampling; no hardware."""

import mujoco as mj
import numpy as np


def body_pair_lever_bound(model, pairs, joint_names):
    """Uniform Lipschitz bound for pair distances vs unknown body angles.

    Rotations ancestral to BOTH geoms cancel from their distance. A rotation
    ancestral to exactly ONE geom changes distance by at most the affected
    point's travel. Its lever is bounded by the sum of kinematic translation
    lengths, local joint/geom offsets, and the enclosing geom sphere radius.
    This triangle bound is valid at every joint angle. Thus max_lever * sum
    absolute body angles covers any sign/order assignment to the named joints.
    """
    joints = [model.joint(name).id for name in joint_names]
    if any(model.jnt_type[j] != mj.mjtJoint.mjJNT_HINGE for j in joints):
        raise ValueError("body uncertainty bound requires revolute joints")

    def lever(joint, geom):
        body = int(model.geom_bodyid[geom])
        ancestor = int(model.jnt_bodyid[joint])
        length = float(np.linalg.norm(model.geom_pos[geom]) + model.geom_rbound[geom])
        while body != ancestor:
            if body == 0:
                return None
            length += float(np.linalg.norm(model.body_pos[body]))
            body = int(model.body_parentid[body])
        if model.geom_type[geom] == mj.mjtGeom.mjGEOM_PLANE or model.geom_plugin[geom] >= 0:
            raise ValueError("cannot bound moving infinite/plugin geometry")
        return length + float(np.linalg.norm(model.jnt_pos[joint]))

    maximum, witness = 0.0, None
    for a, b in pairs:
        for j in joints:
            la, lb = lever(j, a), lever(j, b)
            if (la is None) == (lb is None):
                continue  # unaffected, or a common rigid transform cancels
            value = lb if la is None else la
            if value > maximum:
                maximum = value
                witness = dict(
                    joint=model.joint(j).name,
                    moving_geom=model.geom(a).name,
                    fixed_geom=model.geom(b).name,
                    lever_m=value,
                )
    return dict(
        maximum_pair_lever_m=maximum,
        witness=witness,
        method="common_ancestor_cancellation_and_chain_triangle_bound",
    )


class DistancePairs:
    """Cull only pairs whose enclosing spheres prove distance >= cutoff.

    Planes and plugin geoms always receive a narrow-phase query. The result is
    the same cutoff-clipped minimum as exhaustive mj_geomDistance calls.
    """

    def __init__(self, model, pairs, cutoff=0.15):
        self.pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        self.cutoff = float(cutoff)
        self.radii = model.geom_rbound[self.pairs].sum(axis=1)
        self.uncullable = np.any(
            (model.geom_type[self.pairs] == mj.mjtGeom.mjGEOM_PLANE)
            | (model.geom_plugin[self.pairs] >= 0),
            axis=1,
        )

    def minimum(self, model, data):
        if not len(self.pairs):
            raise ValueError("no geometry pairs")
        a, b = self.pairs.T
        lower = np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b], axis=1) - self.radii
        candidates = np.flatnonzero(self.uncullable | (lower <= self.cutoff + 1e-9))
        best, pair = self.cutoff, None
        for i in candidates:
            aa, bb = int(a[i]), int(b[i])
            distance = float(mj.mj_geomDistance(model, data, aa, bb, self.cutoff, None))
            if not np.isfinite(distance):
                raise ValueError("nonfinite geometry distance")
            if distance < best:
                best, pair = distance, (aa, bb)
        return best, pair
