"""Offline image/pose quality report. Does not solve, qualify, or deploy hand-eye."""

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from cap_harness.realman.calibration import chessboard_pose_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--board", type=Path, default=Path("configs/realman/calibration_board.json")
    )
    args = parser.parse_args()
    board = json.loads(args.board.read_text())
    shape = (board["inner_corners_columns"], board["inner_corners_rows"])
    records, rotations, joints = [], [], []
    for folder in args.samples:
        sample = json.loads((folder / "sample.json").read_text())
        image = cv2.imread(str(folder / "image.png"))
        if image is None:
            raise ValueError(f"missing image: {folder}")
        found, corners = cv2.findChessboardCornersSB(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            shape,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        review = {
            "board_detected": bool(found),
            "handeye_qualified": False,
            "board_frame_identity_verified": False,
        }
        record = {
            "sample": str(folder),
            "arm": sample["arm"],
            "timing_stationarity_passed": sample["timing_stationarity_passed"],
            "corners": len(corners) if found else 0,
            "bracket_span_s": sample["bracket_span_s"],
        }
        if found:
            review["corners_px"] = corners.reshape(-1, 2).tolist()
            review["pose_estimate"] = chessboard_pose_candidates(
                review["corners_px"], sample["camera_info"], board
            )
            record["reprojection_rms_px"] = review["pose_estimate"]["candidates"][0][
                "reprojection_rms_px"
            ]
        (folder / "board_review.json").write_text(json.dumps(review, indent=2) + "\n")
        pose = sample["pose_after"]["arm_state"]["pose"]
        rotations.append(Rotation.from_euler("xyz", np.asarray(pose[3:]) / 1000))
        joints.append(np.asarray(sample["pose_after"]["arm_state"]["joint"]) / 1000)
        records.append(record)
    if len({r["arm"] for r in records}) != 1:
        raise ValueError("never pool separate arm-base frames")
    pairs = []
    for i, j in itertools.combinations(range(len(rotations)), 2):
        rv = (rotations[i].inv() * rotations[j]).as_rotvec()
        pairs.append(
            {
                "i": i,
                "j": j,
                "rotation_deg": float(np.rad2deg(np.linalg.norm(rv))),
                "rotation_vector_rad": rv.tolist(),
            }
        )
    maximum = max((p["rotation_deg"] for p in pairs), default=0)
    span = np.ptp(np.asarray(joints), axis=0).tolist()
    # These are declared collection targets, not universal guarantees of accuracy.
    enough_span = maximum >= 10
    enough_samples = len(records) >= 12
    report = {
        "samples": records,
        "pair_rotations": pairs,
        "joint_span_deg": span,
        "maximum_controller_orientation_span_deg": maximum,
        "collection_targets": {
            "sample_count": 12,
            "orientation_span_deg": 10,
            "nonparallel_rotation_axes_required": True,
        },
        "collection_target_sample_count_met": enough_samples,
        "collection_target_orientation_span_met": enough_span,
        "handeye_qualified": False,
        "transform_written": False,
        "assumptions": [
            "controller RPY treated as fixed XYZ for diagnostics only",
            "board origin identity and independent validation still required",
        ],
        "next_step": "review a larger collision-free multi-axis pose set before collection",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "samples": len(records),
                "max_rotation_deg": maximum,
                "handeye_qualified": False,
                "report": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
