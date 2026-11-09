"""Offline camera-frame measurements from native RGB-D archives.

Manual polygons identify visible surfaces. Fits do not establish material depth
accuracy, hidden geometry, camera extrinsics, or hardware motion targets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .rgbd_geometry import intrinsics, project_depth, zbuffer


def load_cloud(folder):
    folder = Path(folder)
    meta = json.loads((folder / "sample.json").read_text())
    for name in ("color.png", "depth.bin", "depth_m.npy"):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != meta["sha256"][name]:
            raise ValueError("capture checksum mismatch: " + name)
    xyz, uv = project_depth(np.load(folder / "depth_m.npy", allow_pickle=False), meta)
    info = meta["color"]["camera_info"]
    return zbuffer(xyz, uv, info["height"], info["width"]), meta


def polygon_points(cloud, polygons, depth_range=(0.3, 1.0)):
    mask = np.zeros(cloud.shape[:2], np.uint8)
    for polygon in polygons:
        p = np.asarray(polygon, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError("invalid annotation polygon")
        if np.any(p < 0) or np.any(p >= np.array([cloud.shape[1], cloud.shape[0]])):
            raise ValueError("annotation outside image")
        cv2.fillPoly(mask, [np.rint(p).astype(np.int32)], 1)
    valid = np.isfinite(cloud).all(axis=2)
    valid &= (cloud[:, :, 2] >= depth_range[0]) & (cloud[:, :, 2] <= depth_range[1])
    return cloud[(mask > 0) & valid].astype(float)


def fit_plane(points, threshold=0.003, seed=51):
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 30
        or not np.isfinite(points).all()
    ):
        raise ValueError("need at least 30 finite surface points")
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(len(points), 3000), replace=False)]
    best = None
    for _ in range(300):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        offset = -normal @ a
        count = np.count_nonzero(np.abs(sample @ normal + offset) < threshold)
        if best is None or count > best[0]:
            best = count, normal, offset
    if best is None or best[0] < 30:
        raise ValueError("no supported plane")
    normal, offset = best[1:]
    for _ in range(3):
        use = points[np.abs(points @ normal + offset) < threshold]
        if len(use) < 30:
            raise ValueError("plane lost support")
        center = use.mean(axis=0)
        _, singular, directions = np.linalg.svd(use - center, full_matrices=False)
        if singular[1] < 1e-4:
            raise ValueError("degenerate plane patch")
        normal = directions[-1]
        offset = -normal @ center
    if normal[1] > 0:  # camera +Y is down; table normals should point up
        normal, offset = -normal, -offset
    residual = points @ normal + offset
    inside = np.abs(residual) < threshold
    return {
        "normal": normal.tolist(),
        "offset": float(offset),
        "points": len(points),
        "inlier_fraction": float(inside.mean()),
        "inlier_rms_mm": float(np.sqrt(np.mean(residual[inside] ** 2)) * 1000),
        "all_abs_residual_p50_p95_mm": (np.percentile(np.abs(residual), [50, 95]) * 1000).tolist(),
        "threshold_m": threshold,
    }


def intersect_pixels(pixels, info, plane):
    k, distortion = intrinsics(info)
    pixels = np.asarray(pixels, dtype=float)
    rays = cv2.undistortPoints(pixels[:, None], k, distortion)[:, 0]
    rays = np.column_stack([rays, np.ones(len(rays))])
    denominators = rays @ np.asarray(plane["normal"])
    if np.any(np.abs(denominators) < 1e-4):
        raise ValueError("grazing ray cannot constrain a metric point")
    distances = -plane["offset"] / denominators
    if np.any(distances <= 0):
        raise ValueError("plane intersection is behind the camera")
    return rays * distances[:, None]


def measure_segment(pixels, info, plane, pixel_uncertainty=2):
    pixels = np.asarray(pixels, dtype=float)
    if pixels.shape != (2, 2):
        raise ValueError("segment requires two image endpoints")
    xyz = intersect_pixels(pixels, info, plane)
    nominal = float(np.linalg.norm(xyz[1] - xyz[0]))
    distances = []
    shifts = np.array([[x, y] for x in (-1, 1) for y in (-1, 1)]) * pixel_uncertainty
    for first in shifts:
        for second in shifts:
            perturbed = intersect_pixels(pixels + [first, second], info, plane)
            distances.append(float(np.linalg.norm(perturbed[1] - perturbed[0])))
    return {
        "length_m": nominal,
        "endpoints_camera_m": xyz.tolist(),
        "annotation_shift_px": pixel_uncertainty,
        "annotation_only_range_m": [min(distances), max(distances)],
        "metric_accuracy_qualified": False,
    }


def table_basis(plane):
    normal = np.asarray(plane["normal"])
    first = np.array([1.0, 0, 0])
    first -= normal * (first @ normal)
    first /= np.linalg.norm(first)
    return np.array([first, np.cross(normal, first)]).T


def fit_circle_pixels(pixels, info, plane):
    xyz = intersect_pixels(pixels, info, plane)
    basis = table_basis(plane)
    xy = xyz @ basis
    initial = np.r_[xy.mean(axis=0), np.linalg.norm(xy - xy.mean(axis=0), axis=1).mean()]
    residual = lambda v: np.linalg.norm(xy - v[:2], axis=1) - v[2]
    fit = least_squares(residual, initial, loss="soft_l1", f_scale=0.001)
    return {
        "diameter_m": float(2 * fit.x[2]),
        "center_plane_xy_m": fit.x[:2].tolist(),
        "median_abs_residual_mm": float(np.median(np.abs(residual(fit.x))) * 1000),
        "boundary_points": len(pixels),
        "metric_accuracy_qualified": False,
    }


def fit_radial_surface(points, table, *, tapered):
    basis = table_basis(table)
    normal = np.asarray(table["normal"])
    height = points @ normal + table["offset"]
    xy = points @ basis
    if len(points) < 30:
        raise ValueError("insufficient radial surface points")
    middle = np.median(xy, axis=0)
    best = None

    def residual(values):
        radius = values[2] + (values[3] * height if tapered else 0)
        return np.linalg.norm(xy - values[:2], axis=1) - radius

    for direction in (-1, 1):
        for radius in [0.025] if tapered else [0.07, 0.15, 0.3]:
            initial = np.r_[middle + [0, direction * radius], radius]
            lower, upper = np.r_[middle - 0.6, 0.012], np.r_[middle + 0.6, 0.5]
            if tapered:
                initial = np.r_[initial, 0.12]
                lower, upper = np.r_[middle - 0.06, 0.012, 0.01], np.r_[middle + 0.06, 0.045, 0.35]
            fit = least_squares(
                residual, initial, bounds=(lower, upper), loss="soft_l1", f_scale=0.001
            )
            error = float(np.median(np.abs(residual(fit.x))))
            if best is None or error < best[0]:
                best = error, fit.x
    error, parameters = best
    return {
        "center_table_xy_m": parameters[:2].tolist(),
        "radius_at_table_height_m": float(parameters[2]),
        "slope": float(parameters[3]) if tapered else 0.0,
        "surface_median_abs_residual_mm": error * 1000,
        "height_observed_percentiles_m": np.percentile(height, [1, 50, 99]).tolist(),
        "points": len(points),
        "basis_xy": basis.tolist(),
        "qualification": "partial_visible_arc_fit; extrapolated radius requires silhouette cross-check",
    }


def audit_capture(folder, annotations):
    cloud, meta = load_cloud(folder)
    points = lambda polygons: polygon_points(
        cloud, polygons, annotations.get("depth_range_m", [0.3, 1.0])
    )
    table = fit_plane(points(annotations["table"]))
    planes = {"table": table}
    for name, polygons in annotations.get("planes", {}).items():
        planes[name] = fit_plane(points(polygons), threshold=0.002)
    result = {
        "camera": meta["camera"],
        "capture": str(folder),
        "sample_sha256": hashlib.sha256((Path(folder) / "sample.json").read_bytes()).hexdigest(),
        "planes": planes,
        "segments": {},
        "height_regions": {},
        "radial_surfaces": {},
        "circle_regions": {},
        "physical_accuracy_qualified": False,
    }
    for name, spec in annotations.get("segments", {}).items():
        plane = dict(planes[spec.get("plane", "table")])
        if "height_above_table_m" in spec:
            plane["offset"] -= spec["height_above_table_m"]
        result["segments"][name] = measure_segment(
            spec["pixels"], meta["color"]["camera_info"], plane
        )
    for name, polygons in annotations.get("height_regions", {}).items():
        surface = points(polygons)
        if len(surface) < 30:
            result["height_regions"][name] = {"points": len(surface), "insufficient_data": True}
        else:
            heights = surface @ table["normal"] + table["offset"]
            result["height_regions"][name] = {
                "points": len(surface),
                "height_p05_p50_p95_m": np.percentile(heights, [5, 50, 95]).tolist(),
                "full_object_extent": False,
            }
    for name, spec in annotations.get("radial_surfaces", {}).items():
        fitted = fit_radial_surface(points(spec["polygons"]), table, tapered=spec["tapered"])
        if "extent_polygons" in spec:
            surface = points(spec["extent_polygons"])
            heights = surface @ table["normal"] + table["offset"]
            xy = surface @ np.asarray(fitted["basis_xy"])
            radial = np.linalg.norm(xy - fitted["center_table_xy_m"], axis=1)
            predicted = fitted["radius_at_table_height_m"] + fitted["slope"] * heights
            keep = (np.abs(radial - predicted) < 0.003) & (heights > 0) & (heights < 0.2)
            fitted["extent_surface_points"] = int(keep.sum())
            fitted["visible_surface_height_p95_p99_max_m"] = (
                np.percentile(heights[keep], [95, 99, 100]).tolist() if keep.any() else None
            )
            fitted["extent_rule"] = (
                "within 3 mm of fitted radial surface and 0–200 mm above table; not a full-object height certificate"
            )
        result["radial_surfaces"][name] = fitted
    for name, spec in annotations.get("circle_regions", {}).items():
        gray = cv2.imread(str(Path(folder) / "color.png"), cv2.IMREAD_GRAYSCALE)
        x0, y0, x1, y1 = spec["rectangle_xyxy"]
        crop = gray[y0:y1, x0:x1]
        trials = []
        for threshold in spec["gray_thresholds"]:
            contours, _ = cv2.findContours(
                np.uint8(crop < threshold) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                trials.append({"threshold": threshold, "rejected": "no dark component"})
                continue
            contour = max(contours, key=cv2.contourArea)[:, 0]
            if np.any(contour <= 0) or np.any(contour >= [crop.shape[1] - 1, crop.shape[0] - 1]):
                trials.append(
                    {"threshold": threshold, "rejected": "component reaches ROI boundary"}
                )
                continue
            plane = dict(table)
            plane["offset"] -= spec["height_above_table_m"]
            circle = fit_circle_pixels(contour + [x0, y0], meta["color"]["camera_info"], plane)
            circle["threshold"] = threshold
            circle["height_above_table_assumption_m"] = spec["height_above_table_m"]
            trials.append(circle)
        result["circle_regions"][name] = trials
    return result
