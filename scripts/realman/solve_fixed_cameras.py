"""Offline training-only fit, then independent evaluation of frozen camera matrices."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from cap_harness.realman.handeye_solver import pose_difference
from cap_harness.realman.shared_board_calibration import fit_fixed_camera, evaluate_fixed_camera


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows):
    return {key + suffix: float(np.sqrt(np.mean(np.square([r[key] for r in rows]))))
            if suffix == "_rms" else float(max(r[key] for r in rows))
            for key in ["translation_mm", "rotation_deg", "board_center_mm", "reprojection_rms_px"]
            for suffix in ["_rms", "_max"]}


def load_sets(session_path):
    session = json.loads(session_path.read_text())
    sets, seen = [], set()
    for entry in session["accepted"]:
        if entry["path"] in seen or entry["role"] not in ["train", "validation"]:
            raise ValueError("duplicate set or invalid partition")
        seen.add(entry["path"])
        path = session_path.parent / entry["path"] / entry["quality_file"]
        record = json.loads(path.read_text())
        if not record["acquisition_passed"] or record["corner_processing"] != "bounded_single_image_PnP_rejection_v1":
            raise ValueError("unqualified or inconsistent acquisition protocol")
        sets.append({**record, "role": entry["role"], "quality_path": str(path)})
    reference = sets[0]
    for record in sets:
        if record["parent_frame"] != "left_controller_base":
            raise ValueError("wrong base frame")
        for key in ["head", "body"]:
            delta = np.abs(np.asarray(record["fixed_context_millidegrees"][key]) - reference["fixed_context_millidegrees"][key]) / 1000
            if delta.max() > session["acceptance_plan"]["fixed_context_max_joint_change_deg"]:
                raise ValueError("head/body context changed between sets")
        for camera in ["head", "chest"]:
            a, b = record["cameras"][camera]["camera_info"], reference["cameras"][camera]["camera_info"]
            if any(a[k] != b[k] for k in ["width", "height", "k", "d", "distortion_model"]) or a["header"]["frame_id"] != b["header"]["frame_id"]:
                raise ValueError("camera intrinsics/frame changed")
    return session, sets


def camera_records(sets, camera):
    return [{"sample": s["sample"], "Y": np.asarray(s["Y"]),
             "C": np.asarray(s["cameras"][camera]["C"]),
             "objects": np.asarray(s["cameras"][camera]["objects"]),
             "corners": np.asarray(s["cameras"][camera]["corners"])} for s in sets]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", required=True, type=Path)
    p.add_argument("--frozen", required=True, type=Path)
    p.add_argument("--evaluation-output", type=Path)
    args = p.parse_args()
    session, sets = load_sets(args.session)
    training = [s for s in sets if s["role"] == "train"]
    validation = [s for s in sets if s["role"] == "validation"]
    if len(training) < session["minimum_train_poses"]:
        raise ValueError("insufficient training board poses")
    board_path = Path("configs/realman/calibration_board.json")
    board = json.loads(board_path.read_text())
    cols, rows = board["inner_corners_columns"], board["inner_corners_rows"]
    objects = np.zeros((cols * rows, 3))
    objects[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * board["square_size_m"]
    if args.evaluation_output is None:
        if args.frozen.exists():
            raise ValueError("refusing to overwrite frozen training fit")
        pairs = [pose_difference(np.asarray(a["Y"]), np.asarray(b["Y"])) for a, b in itertools.combinations(training, 2)]
        span_deg, span_m = max(v["rotation_deg"] for v in pairs), max(v["translation_mm"] for v in pairs) / 1000
        if span_deg < session["acceptance_plan"]["minimum_board_orientation_span_deg"] or span_m < session["acceptance_plan"]["minimum_board_position_span_m"]:
            raise ValueError("insufficient training board diversity")
        files = [Path(s["quality_path"]) for s in training] + [board_path, Path(session["reference"]),
            Path(session["corner_processing_protocol"]), Path(__file__),
            Path("src/cap_harness/realman/shared_board_calibration.py"),
            Path("src/cap_harness/realman/calibration.py"), Path("scripts/realman/inspect_fixed_camera_pose.py")]
        result = {"status": "frozen_training_fit_awaiting_independent_validation", "qualified": False,
            "parent_frame": "left_controller_base", "training_sets": [s["sample"] for s in training],
            "acceptance_plan": session["acceptance_plan"], "training_orientation_span_deg": span_deg,
            "training_origin_span_m": span_m, "fixed_context_millidegrees": training[0]["fixed_context_millidegrees"],
            "sha256": {str(path): digest(path) for path in files}, "cameras": {}}
        for camera in ["head", "chest"]:
            info = training[0]["cameras"][camera]["camera_info"]
            k, d = np.asarray(info["k"]).reshape(3, 3), np.asarray(info["d"])
            records = camera_records(training, camera)
            fitted = fit_fixed_camera(records, objects, k, d)
            metrics = evaluate_fixed_camera(records, objects, k, d, fitted)
            result["cameras"][camera] = {"T_base_camera": fitted.tolist(), "camera_info": info,
                "training_summary": summary(metrics), "training_samples": metrics}
        args.frozen.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "training": {k:v["training_summary"] for k,v in result["cameras"].items()}}))
        return
    frozen = json.loads(args.frozen.read_text())
    if session["acceptance_plan"] != frozen["acceptance_plan"] or [s["sample"] for s in training] != frozen["training_sets"]:
        raise ValueError("frozen protocol or training partition changed")
    for path, expected in frozen["sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"frozen dependency changed: {path}")
    if len(validation) < session["minimum_validation_poses"]:
        raise ValueError("insufficient independent validation sets")
    limits = frozen["acceptance_plan"]
    result = {"frozen_fit": str(args.frozen), "frozen_fit_sha256": digest(args.frozen),
        "validation_sets": [s["sample"] for s in validation], "refitted": False, "cameras": {}}
    for camera in ["head", "chest"]:
        model = frozen["cameras"][camera]
        info = model["camera_info"]
        metrics = evaluate_fixed_camera(camera_records(validation, camera), objects, np.asarray(info["k"]).reshape(3, 3), np.asarray(info["d"]), np.asarray(model["T_base_camera"]))
        s = summary(metrics)
        gates = {key: s[metric] <= limits[key] for key, metric in [
            ("holdout_origin_rms_mm_max", "translation_mm_rms"), ("holdout_origin_max_mm_max", "translation_mm_max"),
            ("holdout_rotation_rms_deg_max", "rotation_deg_rms"), ("holdout_rotation_max_deg_max", "rotation_deg_max"),
            ("holdout_pixel_rms_px_max", "reprojection_rms_px_rms"), ("holdout_pixel_max_px_max", "reprojection_rms_px_max") ]}
        result["cameras"][camera] = {"summary": s, "samples": metrics, "gates": gates, "passed": all(gates.values())}
    result["numerical_validation_passed"] = all(v["passed"] for v in result["cameras"].values())
    result["qualification_scope"] = "local consistency in left controller base with fixed head/body; not absolute metrology or moving-base calibration"
    args.evaluation_output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
