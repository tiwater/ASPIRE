"""Offline local pose IK; bounded near measured joints, never hardware motion."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from cap_harness.realman.grasp_geometry import grasp_geometry


def solve_local_mount(adapter, arm, target, *, max_delta_rad=0.03, posture_weight=0.001, tilt_weight=1.5):
    m, mj = adapter.model, adapter.mj
    d = mj.MjData(m)
    d.qpos[:] = adapter.data.qpos
    ids = adapter.qadr[arm]
    current = d.qpos[ids].copy()
    bounds = m.jnt_range[adapter.joint_ids[arm]]
    low = np.maximum(bounds[:, 0] + 1e-7, current - max_delta_rad)
    high = np.minimum(bounds[:, 1] - 1e-7, current + max_delta_rad)
    if np.any(low >= high):
        raise ValueError("no local IK interval")
    sid = m.site(arm + "_tool_mount").id

    offset=np.array(grasp_geometry(m,adapter.data,arm)["T_mount_grasp"])[:3,3]
    grasp_target=target[:3,3]+target[:3,:3]@offset

    torso = m.geom("xb_link_collision_0").id
    near = [m.geom(f"l_link_{i}_collision_0").id for i in range(2, 7)] if arm == "left" else []

    def residual(q):
        d.qpos[ids] = q
        mj.mj_kinematics(m, d)
        p = d.site_xpos[sid] + d.site_xmat[sid].reshape(3,3)@offset - grasp_target
        r = Rotation.from_matrix(target[:3, :3] @ d.site_xmat[sid].reshape(3, 3).T).as_rotvec()
        clearance = [max(0.0, 0.002 - mj.mj_geomDistance(m, d, torso, g, 0.03, None)) for g in near]
        return np.r_[50 * p, tilt_weight * r[:2], 10 * r[2:3], posture_weight * (q - current), 200 * np.array(clearance), ([.01*(q[2]-2.3)] if arm=='left' else [])]

    fit = least_squares(
        residual,
        np.clip(current, low + 1e-9, high - 1e-9),
        bounds=(low, high),
        max_nfev=200,
        diff_step=1e-5,
        gtol=1e-10,
        ftol=1e-10,
        xtol=1e-10,
    )
    residual(fit.x)
    pe = float(np.linalg.norm(d.site_xpos[sid] + d.site_xmat[sid].reshape(3,3)@offset - grasp_target))
    re = float(Rotation.from_matrix(target[:3, :3] @ d.site_xmat[sid].reshape(3, 3).T).magnitude())
    rot_error=Rotation.from_matrix(target[:3,:3]@d.site_xmat[sid].reshape(3,3).T).as_rotvec()
    actual_clearance=min((float(mj.mj_geomDistance(m,d,torso,g,.03,None)) for g in near),default=float("inf"))
    return {
        "joint_positions": fit.x.tolist(),
        "position_error_m": pe,
        "position_reference": "grasp_center",
        "orientation_error_rad": re,
        "max_joint_delta_rad": float(np.max(np.abs(fit.x - current))),
        "pose_solved": bool(pe < 0.002 and np.linalg.norm(rot_error[:2]) < np.deg2rad(3) and abs(rot_error[2]) < .01 and actual_clearance >= .001),
        "minimum_torso_clearance_m": actual_clearance,
        "world_rotation_error_deg": np.rad2deg(rot_error).tolist(),
        "nfev": fit.nfev,
        "physical_motion_allowed": False,
    }
