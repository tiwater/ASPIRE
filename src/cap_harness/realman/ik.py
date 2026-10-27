"""Offline URDF tool-mount reachability probes. Never a hardware command planner."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def solve_mount_ik(
    adapter, arm: str, position_m, quaternion_wxyz=None, *, seed=0, restarts=8, robot_state=None
):
    """Report both residual and final-state collisions; no path-validity claim.

    With no quaternion, this is a position-only reachability probe. The tool
    mount is not the gripper grasp center. All optimization uses a separate
    MjData instance so the adapter's active state is never altered.
    """
    if arm not in adapter.arms or not 1 <= restarts <= 32:
        raise ValueError("invalid arm/restarts")
    target = np.asarray(position_m, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target must be a finite xyz position in world")
    target_rotation = None
    if quaternion_wxyz is not None:
        q = np.asarray(quaternion_wxyz, dtype=float)
        if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1):
            raise ValueError("need unit wxyz quaternion")
        target_rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    m, mj = adapter.model, adapter.mj
    d = mj.MjData(m)
    d.qpos[:] = adapter.data.qpos
    if robot_state is not None:
        for a in adapter.arms:
            d.qpos[adapter.qadr[a]] = robot_state.joint_positions[a]
    indices = adapter.qadr[arm]
    bounds = m.jnt_range[adapter.joint_ids[arm]]
    sid = m.site(arm + "_tool_mount").id
    rng = np.random.default_rng(seed)
    start_joints = d.qpos[indices].copy()

    def error(q):
        d.qpos[indices] = q
        mj.mj_forward(m, d)
        diff = d.site_xpos[sid] - target
        if target_rotation is None:
            return diff
        angular = Rotation.from_matrix(
            target_rotation @ d.site_xmat[sid].reshape(3, 3).T
        ).as_rotvec()
        return np.r_[diff, 0.15 * angular]

    best = None
    for attempt in range(restarts):
        initial = (
            np.clip(start_joints, bounds[:, 0] + 1e-5, bounds[:, 1] - 1e-5)
            if attempt == 0
            else rng.uniform(bounds[:, 0], bounds[:, 1])
        )
        fit = least_squares(
            error, initial, bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=200, diff_step=1e-5
        )
        residual = float(np.linalg.norm(error(fit.x)))
        if best is None or residual < best[0]:
            best = (residual, fit.x.copy())
        if residual < 1e-5:
            break
    error(best[1])
    position_error = float(np.linalg.norm(d.site_xpos[sid] - target))
    angle_error = (
        None
        if target_rotation is None
        else float(
            np.linalg.norm(
                Rotation.from_matrix(target_rotation @ d.site_xmat[sid].reshape(3, 3).T).as_rotvec()
            )
        )
    )
    chain = set()
    for i in range(m.nbody):
        parent = i
        while parent:
            if m.body(parent).name == arm[0] + "_link_1":
                chain.add(i)
                break
            parent = int(m.body_parentid[parent])
    collisions = []
    for c in d.contact[: d.ncon]:
        if c.dist < -0.0001 and (
            m.geom_bodyid[c.geom1] in chain or m.geom_bodyid[c.geom2] in chain
        ):
            collisions.append(
                {
                    "geom1": m.geom(c.geom1).name,
                    "geom2": m.geom(c.geom2).name,
                    "penetration_m": -float(c.dist),
                }
            )
    return {
        "arm": arm,
        "frame": "world",
        "target_frame": "URDF link_8 tool mount",
        "joint_positions_urdf_rad": best[1].tolist(),
        "position_error_m": position_error,
        "orientation_error_rad": angle_error,
        "pose_solved": position_error < 0.002 and (angle_error is None or angle_error < 0.02),
        "final_state_collisions": collisions,
        "path_checked": False,
        "physical_motion_allowed": False,
    }


class SimulationIKProvider:
    """Explicit backend='realman_sim' for CAP solve_ik, using mounting-frame targets."""

    def __init__(self, adapter):
        self.adapter = adapter

    def solve_ik(self, target_pose, robot_state, *, arm="primary"):
        from cap_harness.contracts import IKResult
        from cap_harness.errors import ApiError, ErrorCode

        if target_pose.frame != "world" or robot_state.embodiment != "realman":
            return IKResult(
                False,
                arm=arm,
                embodiment="realman",
                error=ApiError(
                    ErrorCode.INVALID_REQUEST,
                    "requires realman state and world-frame mounting pose",
                ),
            )
        report = solve_mount_ik(
            self.adapter,
            arm,
            target_pose.position,
            target_pose.quaternion_wxyz,
            robot_state=robot_state,
        )
        ok = report["pose_solved"] and not report["final_state_collisions"]
        return IKResult(
            ok,
            np.array(report["joint_positions_urdf_rad"]) if ok else None,
            arm=arm,
            embodiment="realman",
            diagnostics=report,
            error=None
            if ok
            else ApiError(
                ErrorCode.IK_FAILED, "mount pose unsolved or final-state collision; see diagnostics"
            ),
        )
