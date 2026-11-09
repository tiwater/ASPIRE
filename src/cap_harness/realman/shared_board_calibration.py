"""Offline fixed-camera registration through a calibrated wrist and shared board.

The board may move between sets, but must be stationary within a set. Head/body
pose must remain fixed. This is geometric estimation, never robot actuation.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .calibration import chessboard_pose_candidates, chessboard_pose_with_bounded_rejection
from .handeye_solver import mean_transform, pose_difference, project, transform


def board_pose_upright(gray, camera_info, board, *, robust_corners=False):
    """Fixed raw/normalized detector order; choose by single-image quality only."""
    attempts = []
    for normalize in [False, True]:
        mode = "SB_NORMALIZE_IMAGE" if normalize else "SB_RAW"
        try:
            result = _board_pose_upright_attempt(
                gray, camera_info, board, robust_corners=robust_corners, normalize=normalize
            )
        except ValueError as exc:
            attempts.append({"detector": mode, "error": str(exc)})
            # Preserve strict-mode fallback only for a missing complete board.
            if normalize or (not robust_corners and str(exc) != "full board not detected"):
                raise
        else:
            result["detector"] = mode
            result["detector_attempts"] = attempts + [{"detector": mode, "passed": True}]
            return result


def _board_pose_upright_attempt(gray, camera_info, board, *, robust_corners, normalize):
    """For visually reviewed upright views of the same physical board face only.

    Origin is the physically upper-left inner corner. The caller must establish
    that image-up corresponds to board-up; an upside-down camera is not supported.
    """
    cols, rows = board["inner_corners_columns"], board["inner_corners_rows"]
    # Reset the seed for reproducible image-local processing; no seed search.
    cv2.setRNGSeed(0)
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    if normalize:
        flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCornersSB(
        gray, (cols, rows), flags=flags
    )
    if not found:
        raise ValueError("full board not detected")
    corners = corners.reshape(-1, 2).astype(float)
    reverse = corners[:cols, 1].mean() > corners[-cols:, 1].mean()
    if reverse:
        corners = corners[::-1].copy()
    grid = corners.reshape(rows, cols, 2)
    horizontal = (grid[:, -1] - grid[:, 0]).mean(axis=0)
    vertical = (grid[-1] - grid[0]).mean(axis=0)
    if horizontal[0] <= abs(horizontal[1]) or vertical[1] <= abs(vertical[0]):
        raise ValueError("view not upright enough for reviewed corner-identity rule")
    objects = np.zeros((cols * rows, 3))
    objects[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * board["square_size_m"]
    if robust_corners:
        quality = chessboard_pose_with_bounded_rejection(corners, camera_info, board)
        ids = quality["retained_ids"]
        return {
            "C": np.asarray(quality["T_camera_board"]),
            "corners": corners[ids], "objects": objects[ids],
            "all_corners": corners, "retained_ids": ids,
            "corner_quality": quality, "reversed": bool(reverse),
            "reprojection_rms_px": quality["retained_rms_px"],
        }
    candidates = chessboard_pose_candidates(corners, camera_info, board)["candidates"]
    if not candidates or candidates[0]["reprojection_rms_px"] > 1:
        raise ValueError("poor board estimate")
    c = np.asarray(candidates[0]["T_camera_board"])
    k, d = np.asarray(camera_info["k"]).reshape(3, 3), np.asarray(camera_info["d"])
    ok, rv, tv = cv2.solvePnP(
        objects,
        corners,
        k,
        d,
        Rotation.from_matrix(c[:3, :3]).as_rotvec(),
        c[:3, 3].copy(),
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("PnP refinement failed")
    c = transform(Rotation.from_rotvec(rv.ravel()).as_matrix(), tv)
    rms = float(np.sqrt(np.mean(np.sum((project(objects, c, k, d) - corners) ** 2, axis=1))))
    return {
        "C": c,
        "corners": corners,
        "objects": objects,
        "reversed": bool(reverse),
        "reprojection_rms_px": rms,
        "all_corners": corners,
        "retained_ids": list(range(len(corners))),
        "corner_quality": None,
    }


def stationary_corner_check(before, after):
    """Check common independently qualified corners, never trim by motion residual.

    Each image's mask is fixed by single-image PnP before this comparison. The
    union may remove at most four/5% and must preserve 90% spatial coverage.
    A rigid image displacement is not an outlier and must still fail.
    """
    a, b = np.asarray(before["all_corners"]), np.asarray(after["all_corners"])
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("inconsistent complete corner arrays")
    ids = np.intersect1d(before["retained_ids"], after["retained_ids"])
    excluded = sorted(set(range(len(a))) - set(ids.tolist()))
    if len(excluded) > min(4, len(a) // 20):
        raise ValueError("too few common qualified corners")
    hull_fractions = []
    for points in [a, b]:
        whole = cv2.contourArea(cv2.convexHull(points.astype(np.float32)))
        kept = cv2.contourArea(cv2.convexHull(points[ids].astype(np.float32)))
        if whole <= 0 or kept / whole < 0.9:
            raise ValueError("common corners lack board-wide coverage")
        hull_fractions.append(kept / whole)
    raw = np.linalg.norm(a - b, axis=1)
    delta = raw[ids]
    result = {
        "common_ids": ids.tolist(), "excluded_union_ids": excluded,
        "common_count": len(ids), "common_hull_fractions": hull_fractions,
        "raw_motion_rms_px": float(np.sqrt(np.mean(raw**2))),
        "raw_motion_max_px": float(raw.max()),
        "motion_rms_px": float(np.sqrt(np.mean(delta**2))),
        "motion_max_px": float(delta.max()),
    }
    if result["motion_rms_px"] > 0.5 or result["motion_max_px"] > 1:
        raise ValueError("board/reference changed within set")
    return result


def fixed_camera_candidate(base_T_board, camera_T_board):
    return base_T_board @ np.linalg.inv(camera_T_board)


def fit_fixed_camera(records, objects, k, d):
    """Fit ONLY explicitly supplied training records. Validate separately."""
    if len(records) < 3:
        raise ValueError("multiple distinct board poses required")
    base_poses = [r["Y"] for r in records]
    translations = np.array([p[:3, 3] for p in base_poses])
    if np.max(np.linalg.norm(translations - translations[0], axis=1)) < 0.03:
        raise ValueError("insufficient board-position diversity")
    initial = mean_transform([fixed_camera_candidate(r["Y"], r["C"]) for r in records])

    def unpack(v):
        return transform(Rotation.from_rotvec(v[:3]).as_matrix(), v[3:])

    def residual(v):
        inverse = np.linalg.inv(unpack(v))
        return np.concatenate(
            [(project(r.get("objects", objects), inverse @ r["Y"], k, d) - r["corners"]).ravel() for r in records]
        )

    initial_vector = np.r_[Rotation.from_matrix(initial[:3, :3]).as_rotvec(), initial[:3, 3]]
    fitted = least_squares(
        residual, initial_vector, max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-10
    )
    if not fitted.success:
        raise ValueError("fixed-camera fit did not converge")
    return unpack(fitted.x)


def evaluate_fixed_camera(records, objects, k, d, base_T_camera):
    center = np.r_[objects.mean(axis=0), 1]
    results = []
    for r in records:
        observed = base_T_camera @ r["C"]
        e = pose_difference(observed, r["Y"])
        e["board_center_mm"] = float(
            1000 * np.linalg.norm((observed @ center)[:3] - (r["Y"] @ center)[:3])
        )
        predicted = project(r.get("objects", objects), np.linalg.inv(base_T_camera) @ r["Y"], k, d)
        e["reprojection_rms_px"] = float(
            np.sqrt(np.mean(np.sum((predicted - r["corners"]) ** 2, axis=1)))
        )
        results.append({"sample": r["sample"], **e})
    return results
