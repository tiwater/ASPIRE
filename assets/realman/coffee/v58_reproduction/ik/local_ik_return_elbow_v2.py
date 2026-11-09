"""Offline pose IK with explicit torso clearance; does not modify live state."""

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.spatial.transform import Rotation


def solve_local_mount(adapter, arm, target, *, max_delta_rad=0.03, posture_weight=0.001):
    m, mj = adapter.model, adapter.mj
    # Single-controller scratch space; only copied qpos and derived kinematics.
    d = getattr(adapter, "_pour_ik_scratch", None)
    if d is None:
        d = mj.MjData(m)
        adapter._pour_ik_scratch = d
    d.qpos[:] = adapter.data.qpos
    ids = adapter.qadr[arm]
    seed = d.qpos[ids].copy()
    bounds = m.jnt_range[adapter.joint_ids[arm]].copy()
    if arm == "left":
        # Retain the observed elbow branch while the wrist reverses through zero.
        # Same model joint limits, 0.03-rad increment and collision checks apply.
        # A measured servo residual may start outside this planning envelope;
        # allow correction only within the unchanged joint-step interval.
        bounds[2, 0] = max(bounds[2, 0], 1.9)
        bounds[2, 1] = min(bounds[2, 1], 2.6)
        if seed[2] < bounds[2, 0] - max_delta_rad or seed[2] > bounds[2, 1] + max_delta_rad:
            raise ValueError("return elbow branch must be entered before wrist reversal")
    bounds[:, 0] = np.maximum(bounds[:, 0] + 1e-7, seed - max_delta_rad)
    bounds[:, 1] = np.minimum(bounds[:, 1] - 1e-7, seed + max_delta_rad)
    sid = m.site(arm + "_tool_mount").id
    torso = m.geom("xb_link_collision_0").id
    near = [m.geom(f"{arm[0]}_link_{i}_collision_0").id for i in range(2, 7)]

    def equality(q):
        d.qpos[ids] = q
        mj.mj_kinematics(m, d)
        return np.r_[
            100 * (d.site_xpos[sid] - target[:3, 3]),
            Rotation.from_matrix(target[:3, :3] @ d.site_xmat[sid].reshape(3, 3).T).as_rotvec(),
        ]

    def clearance(q):
        d.qpos[ids] = q
        mj.mj_kinematics(m, d)
        return (
            np.array([mj.mj_geomDistance(m, d, torso, g, 0.05, None) - 0.004 for g in near]) * 100
        )

    def objective(q):
        return 0.01 * np.sum((q - seed) ** 2) + (
            posture_weight * (q[2] - 2.3) ** 2 if arm == "left" else 0.0
        )

    fit = minimize(
        objective,
        np.clip(seed, bounds[:, 0] + 1e-9, bounds[:, 1] - 1e-9),
        method="SLSQP",
        bounds=list(map(tuple, bounds)),
        constraints=[{"type": "eq", "fun": equality}, {"type": "ineq", "fun": clearance}],
        options={"maxiter": 120, "ftol": 1e-10},
    )
    fallback = False
    if not fit.success:
        # Independent least-squares seed, retaining the same joint interval and
        # acceptance bounds. No state writes or unbounded joint jumps.
        def residual(q):
            eq = equality(q)
            return np.r_[eq[:3] / 10, 1.5 * eq[3:], np.minimum(clearance(q), 0), 0.001 * (q - seed)]

        candidate = least_squares(
            residual,
            np.clip(seed, bounds[:, 0] + 1e-9, bounds[:, 1] - 1e-9),
            bounds=(bounds[:, 0], bounds[:, 1]),
            max_nfev=200,
            gtol=1e-10,
            ftol=1e-10,
            xtol=1e-10,
        )
        er = equality(candidate.x)
        if (
            np.linalg.norm(er[:3]) / 100 < 0.002
            and np.linalg.norm(er[3:]) < 0.02
            and min(clearance(candidate.x)) >= -0.1
        ):
            fit = candidate
            fallback = True
    err = equality(fit.x)
    gap = float(min(clearance(fit.x)) / 100 + 0.004)
    pe = float(np.linalg.norm(err[:3]) / 100)
    re = float(np.linalg.norm(err[3:]))
    return {
        "joint_positions": fit.x.tolist(),
        "position_error_m": pe,
        "orientation_error_rad": re,
        "minimum_torso_clearance_m": gap,
        "max_joint_delta_rad": float(np.max(np.abs(fit.x - seed))),
        "pose_solved": pe < 0.002 and re < 0.02 and gap >= 0.003,
        "nfev": fit.nfev,
        "optimizer_success": bool(fit.success),
        "least_squares_fallback": fallback,
        "physical_motion_allowed": False,
    }
