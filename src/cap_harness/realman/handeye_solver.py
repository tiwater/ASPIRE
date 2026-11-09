"""Offline eye-in-hand estimation. No control, frame writes, or automatic deployment.

Column-vector convention: G = base_T_tool, C = camera_T_board,
X = tool_T_camera, Y = base_T_board; G @ X @ C = Y.
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation).reshape(3)
    if (
        not np.isfinite(result).all()
        or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(result[:3, :3]), 1, atol=1e-6)
    ):
        raise ValueError("non-finite or non-rigid transform")
    return result


def controller_transform(raw_pose):
    pose = np.asarray(raw_pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("requires six finite controller pose values")
    return transform(Rotation.from_euler("xyz", pose[3:] / 1000).as_matrix(), pose[:3] / 1e6)


def mean_transform(poses):
    poses = np.asarray(poses)
    return transform(
        Rotation.from_matrix(poses[:, :3, :3]).mean().as_matrix(), poses[:, :3, 3].mean(axis=0)
    )


def pose_difference(a, b):
    return {
        "translation_mm": float(1000 * np.linalg.norm(a[:3, 3] - b[:3, 3])),
        "rotation_deg": float(
            np.rad2deg(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())
        ),
    }


def excitation(poses):
    vectors = [
        Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()
        for a, b in itertools.combinations(poses, 2)
    ]
    if not vectors:
        raise ValueError("insufficient poses")
    singular = np.linalg.svd(np.asarray(vectors), compute_uv=False)
    return {
        "maximum_rotation_deg": float(np.rad2deg(max(np.linalg.norm(v) for v in vectors))),
        "rotation_vector_singular_values": singular.tolist(),
        "second_to_first_ratio": float(singular[1] / max(singular[0], 1e-15)),
    }


def solve_closed_form(grippers, boards, method="PARK"):
    if len(grippers) != len(boards) or len(grippers) < 3:
        raise ValueError("requires at least three paired poses")
    e = excitation(grippers)
    if e["maximum_rotation_deg"] < 1 or e["second_to_first_ratio"] < 0.05:
        raise ValueError("insufficient independent rotation excitation")
    if method == "TSAI":
        # OpenCV 4.12 silently returns identity/zero after logging an error when
        # fewer than two pairs survive its 0.3..1.7 quaternion-norm filter.
        # Detect that documented implementation condition before trusting output.
        eligible = 0
        for i, j in itertools.combinations(range(len(grippers)), 2):
            a = np.linalg.inv(grippers[j]) @ grippers[i]
            b = boards[j] @ np.linalg.inv(boards[i])
            norms = [2 * np.sin(Rotation.from_matrix(t[:3, :3]).magnitude() / 2) for t in [a, b]]
            eligible += int(all(0.3 <= n <= 1.7 for n in norms))
        if eligible < 2:
            raise ValueError(
                f"OpenCV Tsai unsupported: only {eligible} eligible relative-rotation pairs; requires >=2"
            )
    r, t = cv2.calibrateHandEye(
        [g[:3, :3] for g in grippers],
        [g[:3, 3] for g in grippers],
        [c[:3, :3] for c in boards],
        [c[:3, 3] for c in boards],
        method=getattr(cv2, "CALIB_HAND_EYE_" + method),
    )
    x = transform(r, t)
    return x, mean_transform([g @ x @ c for g, c in zip(grippers, boards, strict=True)])


def project(objects, camera_T_board, k, d):
    camera_points = objects @ camera_T_board[:3, :3].T + camera_T_board[:3, 3]
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("board behind camera")
    return cv2.projectPoints(
        objects,
        Rotation.from_matrix(camera_T_board[:3, :3]).as_rotvec(),
        camera_T_board[:3, 3],
        k,
        d,
    )[0].reshape(-1, 2)


def refine_pixels(grippers, corners, objects, k, d, x, y, *, objects_per_view=None):
    view_objects = [objects] * len(grippers) if objects_per_view is None else objects_per_view

    def pack(t):
        return np.r_[Rotation.from_matrix(t[:3, :3]).as_rotvec(), t[:3, 3]]

    def unpack(v):
        return transform(Rotation.from_rotvec(v[:3]).as_matrix(), v[3:6])

    def residual(v):
        xx, yy = unpack(v[:6]), unpack(v[6:])
        return np.concatenate(
            [
                (project(o, np.linalg.inv(g @ xx) @ yy, k, d) - c).ravel()
                for g, c, o in zip(grippers, corners, view_objects, strict=True)
            ]
        )

    fit = least_squares(
        residual,
        np.r_[pack(x), pack(y)],
        method="trf",
        loss="linear",
        max_nfev=300,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-10,
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError("pixel refinement did not converge")
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    return (
        unpack(fit.x[:6]),
        unpack(fit.x[6:]),
        {
            "success": bool(fit.success),
            "evaluations": fit.nfev,
            "rms_corner_px": float(np.sqrt(np.mean(fit.fun.reshape(-1, 2) ** 2) * 2)),
            "jacobian_singular_values": singular.tolist(),
        },
    )


def evaluate(records, objects, k, d, x, y):
    results = []
    center = np.r_[objects.mean(axis=0), 1]
    for r in records:
        observed = r["G"] @ x @ r["C"]
        errors = pose_difference(observed, y)
        errors["board_center_mm"] = float(
            1000 * np.linalg.norm((observed @ center)[:3] - (y @ center)[:3])
        )
        predicted = project(r.get("object_points", objects), np.linalg.inv(r["G"] @ x) @ y, k, d)
        errors["reprojection_rms_px"] = float(
            np.sqrt(np.mean(np.sum((predicted - r["corners"]) ** 2, axis=1)))
        )
        results.append({"sample": r["sample"], **errors})
    summary = {}
    for key in ("translation_mm", "rotation_deg", "board_center_mm", "reprojection_rms_px"):
        values = [r[key] for r in results]
        summary[key + "_rms"] = float(np.sqrt(np.mean(np.square(values)))) if values else None
        summary[key + "_max"] = max(values, default=None)
    return {"count": len(results), "summary": summary, "samples": results}
