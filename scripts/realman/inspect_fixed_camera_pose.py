"""Inspect one stationary shared-board set offline; never deploys or moves hardware."""

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

from cap_harness.realman.handeye_solver import controller_transform, mean_transform, pose_difference
from cap_harness.realman.shared_board_calibration import (
    board_pose_upright, fixed_camera_candidate, stationary_corner_check,
)


def payload(path):
    result = json.loads(path.read_text())
    if result.get("isError"):
        raise ValueError(f"failed native acquisition: {path}")
    return json.loads(next(c["text"] for c in result["content"] if c["type"] == "text"))


def stamp(header):
    t = header["stamp"]
    return t["sec"] + t["nanosec"] / 1e9


def inspect(root, board, calibration, *, robust_corners=False):
    if calibration["status"] != "validated_local_geometry" or calibration["arm"] != "left":
        raise ValueError("validated left reference required")
    samples, detections, ys = [], [], []
    for name in ["left_before", "left_after"]:
        receipt = payload(root / (name + ".json"))
        folder = root / Path(receipt["path"]).name
        sample = json.loads((folder / "sample.json").read_text())
        if not sample["timing_stationarity_passed"] or sample["arm"] != "left":
            raise ValueError("invalid wrist bracket")
        for side in ["before", "after"]:
            tool, work = sample["tool_frame_" + side], sample["work_frame_" + side]
            if tool["tool_name"] != "Arm_Tip" or work["frame_name"] != "World" or any(tool["pose"]) or any(work["pose"]):
                raise ValueError("reference frames changed")
        info = sample["camera_info"]
        for key in ["width", "height", "k", "d", "distortion_model"]:
            if info[key] != calibration["camera_info"][key]:
                raise ValueError("reference intrinsics changed")
        if info["header"]["frame_id"] != calibration["child_frame"]:
            raise ValueError("reference optical frame changed")
        found = board_pose_upright(cv2.imread(str(folder / "image.png"), 0), info, board, robust_corners=robust_corners)
        if found["reprojection_rms_px"] > 1:
            raise ValueError("reference PnP exceeds 1px")
        g = mean_transform([controller_transform(sample["pose_" + s]["arm_state"]["pose"]) for s in ["before", "after"]])
        ys.append(g @ np.asarray(calibration["T_tool_camera"]) @ found["C"])
        samples.append(sample)
        detections.append(found)
    corner_stability = stationary_corner_check(*detections)
    board_change = pose_difference(ys[0], ys[1])
    if board_change["translation_mm"] > 1 or board_change["rotation_deg"] > 0.2:
        raise ValueError("board/reference changed within set")
    states = [payload(root / (n + ".json"))["ros"]["state"] for n in ["state_before", "state_after"]]
    fixed_context = {}
    for key in ["body", "head"]:
        change = np.abs(np.asarray(states[0][key]["position"]) - states[1][key]["position"]) / 1000
        if change.max() > 0.05:
            raise ValueError("head/body changed within set")
        fixed_context[key] = states[0][key]["position"]
    for state in states:
        if state["chassis"]["linear"]["x"] != 0 or state["chassis"]["angular"]["z"] != 0:
            raise ValueError("chassis not stationary")
    y = mean_transform(ys)
    result = {"sample": root.name, "acquisition_passed": True, "qualified_extrinsics": False,
              "parent_frame": "left_controller_base", "Y": y.tolist(), "fixed_context_millidegrees": fixed_context,
              "reference_pnp_rms_px": [d["reprojection_rms_px"] for d in detections],
              "corner_processing": "bounded_single_image_PnP_rejection_v1" if robust_corners else "all_measured_corners",
              "reference_corner_motion_rms_px": corner_stability["motion_rms_px"],
              "reference_corner_motion_max_px": corner_stability["motion_max_px"], "reference_board_change": board_change,
              "reference_corner_stability": corner_stability,
              "reference_detections": [{"all_corners": d["all_corners"].tolist(), "corner_quality": d["corner_quality"], "detector": d["detector"], "detector_attempts": d["detector_attempts"]} for d in detections],
              "cameras": {}, "assumptions": ["same physical upright board face visually reviewed", "operator holds board and chassis fixed within each set", "software bracketing, not hardware synchronization"]}
    for camera in ["head", "chest"]:
        path = root / (camera + ".json")
        meta = payload(path)
        info = meta["camera_info"]
        if info["header"]["frame_id"] != camera + "_depth_cam_color_optical_frame":
            raise ValueError("wrong fixed camera frame")
        # cap_observe exposes CameraInfo time and bounded source age; retain both.
        t = stamp(info["header"])
        if not stamp(samples[0]["image"]["header"]) < t < stamp(samples[1]["image"]["header"]):
            raise ValueError("fixed-camera observation outside wrist-image bracket")
        raw = json.loads(path.read_text())
        data = base64.b64decode(next(c["data"] for c in raw["content"] if c["type"] == "image"))
        gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if not (root / (camera + ".png")).exists():
            cv2.imwrite(str(root / (camera + ".png")), gray)
        found = board_pose_upright(gray, info, board, robust_corners=robust_corners)
        if found["reprojection_rms_px"] > 1:
            raise ValueError("fixed camera PnP exceeds 1px")
        result["cameras"][camera] = {"C": found["C"].tolist(), "corners": found["corners"].tolist(),
            "objects": found["objects"].tolist(), "camera_info": info,
            "all_corners": found["all_corners"].tolist(), "retained_ids": found["retained_ids"],
            "corner_quality": found["corner_quality"],
            "detector": found["detector"], "detector_attempts": found["detector_attempts"],
            "pnp_rms_px": found["reprojection_rms_px"], "source_age_s": meta["source_age_s"],
            "T_base_camera_candidate": fixed_camera_candidate(y, found["C"]).tolist()}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose", required=True, type=Path)
    p.add_argument("--board", type=Path, default=Path("configs/realman/calibration_board.json"))
    p.add_argument("--reference", type=Path, default=Path("configs/realman/calibration/left_wrist_20261027.json"))
    p.add_argument("--robust-corners", action="store_true", help="Use bounded single-image corner rejection uniformly")
    args = p.parse_args()
    result = inspect(args.pose, json.loads(args.board.read_text()), json.loads(args.reference.read_text()), robust_corners=args.robust_corners)
    output = "quality_robust.json" if args.robust_corners else "quality.json"
    (args.pose / output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"acquisition_passed": True, "reference_motion_rms_px": result["reference_corner_motion_rms_px"],
        "reference_pnp_rms_px": result["reference_pnp_rms_px"],
        "fixed_camera_pnp_rms_px": {k:v["pnp_rms_px"] for k,v in result["cameras"].items()}}))


if __name__ == "__main__":
    main()
