"""Solve an explicitly partitioned stationary dataset, with untouched holdout poses.

Writes a report only, never a deployed calibration or controller parameter.
"""

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from cap_harness.realman.calibration import (
    chessboard_pose_candidates,
    chessboard_pose_with_bounded_rejection,
)
from cap_harness.realman.handeye_solver import (
    controller_transform,
    evaluate,
    excitation,
    mean_transform,
    pose_difference,
    refine_pixels,
    solve_closed_form,
    transform,
)


def load_records(root, manifest, board, arm="left", *, robust_corners=False):
    if arm not in ("left", "right"):
        raise ValueError("unknown arm")
    records, previous, reference_info = [], None, None
    shape = (board["inner_corners_columns"], board["inner_corners_rows"])
    objects = np.zeros((shape[0] * shape[1], 3))
    objects[:, :2] = np.mgrid[0 : shape[0], 0 : shape[1]].T.reshape(-1, 2) * board["square_size_m"]
    ordered = sorted(
        manifest,
        key=lambda r: json.loads((root / Path(r["path"]).name / "sample.json").read_text())[
            "started_unix_s"
        ],
    )
    if len({Path(r["path"]).name for r in ordered}) != len(ordered):
        raise ValueError("duplicate sample in manifest")
    for entry in ordered:
        folder = root / Path(entry["path"]).name
        sample = json.loads((folder / "sample.json").read_text())
        if not sample["timing_stationarity_passed"] or sample["arm"] != arm:
            raise ValueError("unqualified acquisition or wrong arm")
        for side in ["before", "after"]:
            tool, work = sample["tool_frame_" + side], sample["work_frame_" + side]
            if (
                tool["tool_name"] != "Arm_Tip"
                or work["frame_name"] != "World"
                or any(tool["pose"])
                or any(work["pose"])
            ):
                raise ValueError("changed tool/work frame")
        info = sample["camera_info"]
        if info["header"]["frame_id"] != arm + "_arm_depth_cam_color_optical_frame":
            raise ValueError("camera optical frame does not match requested arm")
        signature = {key: info[key] for key in ["width", "height", "k", "d", "distortion_model"]}
        signature["frame_id"] = info["header"]["frame_id"]
        if reference_info is None:
            reference_info = signature
        if signature != reference_info:
            raise ValueError("camera/intrinsics changed")
        gray = cv2.imread(str(folder / "image.png"), cv2.IMREAD_GRAYSCALE)
        found, corners = cv2.findChessboardCornersSB(
            gray, shape, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        detector = "SB_EXHAUSTIVE_ACCURACY"
        if not found and robust_corners:
            found, corners = cv2.findChessboardCornersSB(
                gray,
                shape,
                flags=cv2.CALIB_CB_EXHAUSTIVE
                | cv2.CALIB_CB_ACCURACY
                | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            detector += "_NORMALIZE_IMAGE"
        if not found:
            raise ValueError(f"board missing: {folder}")
        corners = corners.reshape(-1, 2).astype(float)
        flipped = False
        if previous is not None:
            distances = [
                np.linalg.norm(corners - previous),
                np.linalg.norm(corners[::-1] - previous),
            ]
            if min(distances) * 3 > max(distances):
                raise ValueError("ambiguous board corner identity")
            if distances[1] < distances[0]:
                corners, flipped = corners[::-1].copy(), True
            if np.max(np.linalg.norm(corners - previous, axis=1)) > 200:
                raise ValueError("board tracking jump needs review")
        previous = corners
        candidates = chessboard_pose_candidates(corners, info, board)["candidates"]
        if not candidates or (not robust_corners and candidates[0]["reprojection_rms_px"] > 1):
            raise ValueError("poor planar pose estimate")
        c = np.asarray(candidates[0]["T_camera_board"])
        k, d = np.asarray(info["k"]).reshape(3, 3), np.asarray(info["d"])
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
        full_corners = corners.copy()
        point_objects = objects
        corner_quality = None
        if robust_corners:
            corner_quality = chessboard_pose_with_bounded_rejection(corners, info, board)
            ids = corner_quality["retained_ids"]
            c = np.asarray(corner_quality["T_camera_board"])
            corners, point_objects = corners[ids], objects[ids]
        g = mean_transform(
            [
                controller_transform(sample["pose_" + side]["arm_state"]["pose"])
                for side in ["before", "after"]
            ]
        )
        records.append(
            {
                "sample": folder.name,
                "detector": detector,
                "role": entry["role"],
                "G": g,
                "C": c,
                "corners": corners,
                "all_detected_corners": full_corners,
                "object_points": point_objects,
                "corner_quality": corner_quality,
                "corner_order_reversed": flipped,
                "pnp_rms_px": corner_quality["retained_rms_px"]
                if robust_corners
                else candidates[0]["reprojection_rms_px"],
            }
        )
    return records, objects, k, d, reference_info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=("left", "right"), default="left")
    p.add_argument(
        "--robust-corners",
        action="store_true",
        help="Opt in to fixed single-image rejection of <=5%% and <=4 corners; retain audit masks",
    )
    p.add_argument("--board", type=Path, default=Path("configs/realman/calibration_board.json"))
    args = p.parse_args()
    records, objects, k, d, info = load_records(
        args.manifest.parent,
        json.loads(args.manifest.read_text()),
        json.loads(args.board.read_text()),
        args.arm,
        robust_corners=args.robust_corners,
    )
    training = [r for r in records if r["role"] == "train"]
    validation = [r for r in records if r["role"] == "validation"]
    if len(training) < 12 or len(validation) < 4 or len(training) + len(validation) != len(records):
        raise ValueError("requires >=12 training and >=4 held-out records, explicit roles")
    gs, cs = [r["G"] for r in training], [r["C"] for r in training]
    methods, solutions = {}, {}
    for method in ["PARK", "TSAI", "HORAUD", "ANDREFF", "DANIILIDIS"]:
        try:
            x, y = solve_closed_form(gs, cs, method)
            solutions[method] = (x, y)
            methods[method] = {
                "T_tool_camera": x.tolist(),
                "training": evaluate(training, objects, k, d, x, y)["summary"],
            }
        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
            methods[method] = {"error": str(exc)}
    # Select a method a priori; held-out poses never choose or refine the fit.
    x0, y0 = solutions["PARK"]
    x, y, optimization = refine_pixels(
        gs,
        [r["corners"] for r in training],
        objects,
        k,
        d,
        x0,
        y0,
        objects_per_view=[r["object_points"] for r in training],
    )
    train_result = evaluate(training, objects, k, d, x, y)
    validation_result = evaluate(validation, objects, k, d, x, y)
    # Solver availability is determined from training closure, never holdout.
    # Preserve failed candidates in the report; require three mutually agreeing
    # successful methods. A silent failure returning identity is not a solution.
    accepted = []
    for method, result in methods.items():
        quality = result.get("training")
        valid = (
            quality is not None
            and quality["translation_mm_rms"] <= 3
            and quality["rotation_deg_rms"] <= 0.5
            and quality["reprojection_rms_px_rms"] <= 1.5
        )
        result["training_consistency_passed"] = bool(valid)
        if valid:
            accepted.append(method)
    agreements = {
        a + "_" + b: pose_difference(solutions[a][0], solutions[b][0])
        for a, b in itertools.combinations(accepted, 2)
    }
    leave_one_out = []
    for index in range(len(training)):
        subset = training[:index] + training[index + 1 :]
        gg, cc = [r["G"] for r in subset], [r["C"] for r in subset]
        xx, yy = solve_closed_form(gg, cc)
        xx, yy, _ = refine_pixels(
            gg,
            [r["corners"] for r in subset],
            objects,
            k,
            d,
            xx,
            yy,
            objects_per_view=[r["object_points"] for r in subset],
        )
        leave_one_out.append({"omitted": training[index]["sample"], **pose_difference(xx, x)})
    e = excitation(gs)
    v = validation_result["summary"]
    gates = {
        "rotation_span_at_least_10deg": e["maximum_rotation_deg"] >= 10,
        "independent_axes_ratio_at_least_0_2": e["second_to_first_ratio"] >= 0.2,
        "holdout_origin_rms_at_most_3mm": v["translation_mm_rms"] <= 3,
        "holdout_origin_max_at_most_5mm": v["translation_mm_max"] <= 5,
        "holdout_rotation_rms_at_most_0_5deg": v["rotation_deg_rms"] <= 0.5,
        "holdout_rotation_max_at_most_1deg": v["rotation_deg_max"] <= 1,
        "holdout_pixel_rms_at_most_1_5px": v["reprojection_rms_px_rms"] <= 1.5,
        "holdout_pixel_max_at_most_2px": v["reprojection_rms_px_max"] <= 2,
        "closed_form_agreement_at_most_10mm_2deg": all(
            v["translation_mm"] <= 10 and v["rotation_deg"] <= 2 for v in agreements.values()
        ),
        "at_least_three_training_consistent_methods": len(accepted) >= 3,
        "leave_one_out_at_most_10mm_2deg": all(
            v["translation_mm"] <= 10 and v["rotation_deg"] <= 2 for v in leave_one_out
        ),
        "wrist_camera_translation_below_0_5m": np.linalg.norm(x[:3, 3]) < 0.5,
    }
    report = {
        "schema_version": 1,
        "arm": args.arm,
        "corner_processing": "bounded_single_image_PnP_rejection_v1"
        if args.robust_corners
        else "all_detected_corners",
        "pixel_metrics_scope": "retained_measured_corners; board center uses full board geometry",
        "parent_frame": args.arm + "_controller_Arm_Tip",
        "child_frame": info["frame_id"],
        "equation": "base_T_tool @ tool_T_camera @ camera_T_board = base_T_board",
        "pose_convention": "controller micrometres / milliradians; fixed XYZ RPY = Rz(yaw) Ry(pitch) Rx(roll)",
        "T_tool_camera": x.tolist(),
        "T_base_board": y.tolist(),
        "camera_info": info,
        "methods": methods,
        "method_agreement": agreements,
        "pixel_refinement": optimization,
        "accepted_comparison_methods": accepted,
        "refinement_vs_PARK": pose_difference(x, x0),
        "excitation": e,
        "training": train_result,
        "validation": validation_result,
        "leave_one_out": leave_one_out,
        "gates": {key: bool(value) for key, value in gates.items()},
        "numerical_validation_passed": bool(all(gates.values())),
        "physical_qualification": "requires separate fixed-board and mounting review; local validation is not absolute metrology",
        "deployed": False,
        "manifest": str(args.manifest),
        "records": [
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in r.items()
            }
            for r in records
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["numerical_validation_passed"],
                "gates": report["gates"],
                "validation": v,
                "X": x.tolist(),
                "excitation": e,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
