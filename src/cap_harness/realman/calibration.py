"""Read-only board/intrinsics readiness checks; no robot motion or auto-qualification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .capture import image_array


def board_spec(config):
    kind = config.get("type")
    if kind is None:
        return None
    if kind not in ("chessboard", "charuco"):
        raise ValueError(
            "supported boards: chessboard or charuco; other patterns need their exact definition"
        )
    prefix = "inner_corners" if kind == "chessboard" else "charuco_squares"
    cols, rows = config[prefix + "_columns"], config[prefix + "_rows"]
    if type(cols) is not int or type(rows) is not int or min(cols, rows) < 3:
        raise ValueError("board rows/columns must be integers >= 3")
    size = config.get("square_size_m")
    if not isinstance(size, (float, int)) or not np.isfinite(size) or not 0 < size < 1:
        raise ValueError("physical square_size_m is required")
    if kind == "charuco":
        marker = config.get("marker_size_m")
        if (
            not isinstance(marker, (float, int))
            or not 0 < marker < size
            or not config.get("aruco_dictionary")
        ):
            raise ValueError("ChArUco needs dictionary and marker size smaller than square size")
    return kind, cols, rows, float(size)


def chessboard_pose_candidates(corners_px, camera_info, board_config):
    """Estimate board->optical-camera poses using existing, unverified intrinsics.

    Keep both planar PnP solutions. Small reprojection residual is not proof of
    metric accuracy or hand-eye calibration. Origin follows detector corner order.
    """
    import cv2

    spec = board_spec(board_config)
    if spec is None or spec[0] != "chessboard":
        raise ValueError("requires a known chessboard")
    _, cols, rows, size = spec
    points = np.asarray(corners_px, dtype=float)
    k = np.asarray(camera_info["k"], dtype=float).reshape(3, 3)
    d = np.asarray(camera_info["d"], dtype=float)
    if camera_info.get("distortion_model") not in ("plumb_bob", "rational_polynomial"):
        raise ValueError("unsupported distortion model for pinhole PnP")
    if (
        points.shape != (cols * rows, 2)
        or not np.isfinite(points).all()
        or not np.isfinite(k).all()
        or min(k[0, 0], k[1, 1]) <= 0
        or d.ndim != 1
        or len(d) not in (4, 5, 8, 12, 14)
        or not np.isfinite(d).all()
    ):
        raise ValueError("invalid corner/intrinsics data")
    objects = np.zeros((cols * rows, 3), dtype=float)
    objects[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * size
    count, rotations, translations, _ = cv2.solvePnPGeneric(
        objects, points, k, d, flags=cv2.SOLVEPNP_IPPE
    )
    candidates = []
    for rvec, tvec in zip(rotations, translations, strict=True):
        rotation = cv2.Rodrigues(rvec)[0]
        camera_points = objects @ rotation.T + tvec.reshape(1, 3)
        if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
            continue
        projected = cv2.projectPoints(objects, rvec, tvec, k, d)[0].reshape(-1, 2)
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, tvec.ravel()
        center = rotation @ objects.mean(axis=0) + tvec.ravel()
        candidates.append(
            {
                "T_camera_board": transform.tolist(),
                "board_center_camera_m": center.tolist(),
                "reprojection_rms_px": float(
                    np.sqrt(np.mean(np.sum((projected - points) ** 2, axis=1)))
                ),
            }
        )
    candidates.sort(key=lambda item: item["reprojection_rms_px"])
    return {
        "method": "IPPE_with_existing_CameraInfo",
        "solver_solution_count": count,
        "candidates": candidates,
        "metric_accuracy_verified": False,
        "handeye_qualified": False,
        "board_frame": "detector_first_inner_corner; origin/direction must be tracked across views",
        "corner_hull_image_fraction": float(
            cv2.contourArea(cv2.convexHull(points.astype(np.float32)))
        )
        / (camera_info["width"] * camera_info["height"]),
    }


def chessboard_pose_with_bounded_rejection(corners_px, camera_info, board_config):
    """Fit measured corners, excluding at most 5% (and at most four) outliers.

    The fixed 2 px rule uses only this image's PnP residual, never robot poses,
    hand-eye closure or validation scores. No pixels/corners are synthesized.
    Failed or ambiguous frames must be reacquired, not progressively trimmed.
    This is an opt-in acquisition rule, not a qualification of metric accuracy.
    """
    import cv2

    initial = chessboard_pose_candidates(corners_px, camera_info, board_config)
    if not initial["candidates"]:
        raise ValueError("no positive-depth planar pose")
    _, cols, rows, size = board_spec(board_config)
    points = np.asarray(corners_px, dtype=float)
    objects = np.zeros((cols * rows, 3), dtype=float)
    objects[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * size
    k = np.asarray(camera_info["k"], dtype=float).reshape(3, 3)
    d = np.asarray(camera_info["d"], dtype=float)
    pose = np.asarray(initial["candidates"][0]["T_camera_board"])
    rv, tv = cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3].copy()
    keep = np.ones(len(points), dtype=bool)
    maximum_excluded = min(4, len(points) // 20)
    full_fit_rms = None
    for _ in range(maximum_excluded + 2):
        ok, rv, tv = cv2.solvePnP(
            objects[keep],
            points[keep],
            k,
            d,
            rv,
            tv,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise ValueError("PnP refinement failed")
        predicted = cv2.projectPoints(objects, rv, tv, k, d)[0].reshape(-1, 2)
        residuals = np.linalg.norm(predicted - points, axis=1)
        if not np.isfinite(residuals).all():
            raise ValueError("nonfinite PnP residual")
        if full_fit_rms is None:
            full_fit_rms = float(np.sqrt(np.mean(residuals**2)))
        next_keep = keep & (residuals <= 2.0)
        if np.count_nonzero(~next_keep) > maximum_excluded:
            raise ValueError("too many outlier corners; reacquire image")
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    else:
        raise ValueError("outlier mask did not converge")
    rms = float(np.sqrt(np.mean(residuals[keep] ** 2)))
    if rms > 1.0:
        raise ValueError("retained corner RMS exceeds 1 px")
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
    if np.any((objects @ pose[:3, :3].T + pose[:3, 3])[:, 2] <= 0):
        raise ValueError("board behind camera")
    # Preserve board-wide coverage; do not accept a tiny surviving region.
    hull = cv2.contourArea(cv2.convexHull(points.astype(np.float32)))
    kept_hull = cv2.contourArea(cv2.convexHull(points[keep].astype(np.float32)))
    if hull <= 0 or kept_hull / hull < 0.9:
        raise ValueError("insufficient retained corner coverage")
    return {
        "method": "bounded_single_image_PnP_rejection_v1",
        "T_camera_board": pose.tolist(),
        "retained_ids": np.flatnonzero(keep).tolist(),
        "excluded_ids": np.flatnonzero(~keep).tolist(),
        "outlier_threshold_px": 2.0,
        "maximum_excluded": maximum_excluded,
        "initial_ippe_rms_px": initial["candidates"][0]["reprojection_rms_px"],
        "full_fit_rms_px": full_fit_rms,
        "retained_rms_px": rms,
        "retained_max_px": float(residuals[keep].max()),
        "all_corner_residuals_px": residuals.tolist(),
        "retained_hull_fraction": kept_hull / hull,
        "metric_accuracy_verified": False,
        "handeye_qualified": False,
    }


def inspect_capture(folder: Path, board_config: dict):
    raw = json.loads((folder / "capture.json").read_text())
    spec = board_spec(board_config)
    reports = {}
    for stream, meta in raw["images"].items():
        if meta["encoding"] not in ("rgb8", "bgr8"):
            continue
        rgb = image_array(folder, meta)
        info = raw["camera_info"].get(stream, {}).get("message")
        ready = False
        if info is not None:
            k = np.asarray(info["k"], dtype=float).reshape(3, 3)
            d = np.asarray(info["d"], dtype=float)
            ready = bool(
                np.isfinite(k).all()
                and np.isfinite(d).all()
                and k[0, 0] > 0
                and k[1, 1] > 0
                and info["width"] == meta["width"]
                and info["height"] == meta["height"]
                and info["header"]["frame_id"] == meta["header"]["frame_id"]
            )
        report = {
            "optical_frame": meta["header"]["frame_id"],
            "intrinsics_metadata_consistent": ready,
            "distortion_model": info.get("distortion_model") if info else None,
            "board_spec_available": spec is not None,
            "board_detected": None,
            "intrinsics_accuracy_verified": False,
            "handeye_qualified": False,
        }
        if spec is not None:
            import cv2

            kind, cols, rows, size = spec
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if kind == "chessboard":
                ok, corners = cv2.findChessboardCornersSB(
                    gray, (cols, rows), cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                ids = None
            else:
                name = board_config["aruco_dictionary"]
                if not name.startswith("DICT_") or not hasattr(cv2.aruco, name):
                    raise ValueError("unknown OpenCV ArUco dictionary")
                dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
                board = cv2.aruco.CharucoBoard(
                    (cols, rows), size, board_config["marker_size_m"], dictionary
                )
                corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
                ok = corners is not None and len(corners) >= 6
            report["board_detected"] = bool(ok)
            report["corners_px"] = corners.reshape(-1, 2).tolist() if ok else []
            report["corner_ids"] = ids.flatten().tolist() if ok and ids is not None else None
            if ok and ready and kind == "chessboard":
                try:
                    report["pose_estimate"] = chessboard_pose_candidates(
                        corners.reshape(-1, 2), info, board_config
                    )
                except (ValueError, cv2.error) as exc:
                    report["pose_estimate_error"] = str(exc)
        reports[stream] = report
    return {
        "capture": str(folder),
        "capture_started_unix_s": raw["started_unix_s"],
        "mode": "offline_readonly_precheck",
        "cameras": reports,
        "missing": ["board specification"] if spec is None else [],
        "controller_pose_pairs_available": False,
        "handeye_qualified": False,
        "note": "Corner detection and valid CameraInfo are prerequisites, not calibration results. Joint-only diagnostic samples are not hand-eye pose pairs.",
    }
