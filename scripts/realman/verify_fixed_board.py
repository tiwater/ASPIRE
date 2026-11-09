"""Compare fixed-camera checkpoints; observation evidence, not continuous tracking."""

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", nargs="+", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    images, infos, reference, corners = [], [], None, None
    for index, path in enumerate(args.observations):
        result = json.loads(path.read_text())
        blob = base64.b64decode(next(c["data"] for c in result["content"] if c["type"] == "image"))
        im = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(path.with_suffix(".png")), im)
        meta = json.loads(next(c["text"] for c in result["content"] if c["type"] == "text"))
        infos.append(meta["camera_info"])
        images.append(im)
        if reference is None:
            ok, detected = cv2.findChessboardCornersSB(
                im,
                (11, 8),
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_EXHAUSTIVE
                | cv2.CALIB_CB_ACCURACY,
            )
            if ok:
                reference, corners = index, detected.astype(np.float32)
    if reference is None:
        raise ValueError("no full chessboard checkpoint")
    frame_ids = {info["header"]["frame_id"] for info in infos}
    if len(frame_ids) != 1 or any(info["k"] != infos[0]["k"] for info in infos):
        raise ValueError("fixed camera changed")
    records = []
    for path, im in zip(args.observations, images, strict=True):
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            images[reference], im, corners, None, winSize=(21, 21), maxLevel=3
        )
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            im, images[reference], moved, None, winSize=(21, 21), maxLevel=3
        )
        fb = np.linalg.norm((back - corners).reshape(-1, 2), axis=1)
        valid = (status.ravel() == 1) & (status_back.ravel() == 1) & (fb < 0.5)
        distance = np.linalg.norm((moved - corners).reshape(-1, 2), axis=1)
        records.append(
            {
                "observation": str(path),
                "tracked_corners": int(valid.sum()),
                "motion_rms_px": float(np.sqrt(np.mean(distance[valid] ** 2))),
                "motion_max_px": float(np.max(distance[valid])),
                "forward_backward_max_px": float(np.max(fb[valid])),
            }
        )
    passed = all(
        r["tracked_corners"] == 88 and r["motion_rms_px"] <= 0.5 and r["motion_max_px"] <= 1
        for r in records
    )
    report = {
        "reference": str(args.observations[reference]),
        "camera_frame": next(iter(frame_ids)),
        "method": "88 chessboard corners tracked bidirectionally from a successfully detected checkpoint",
        "checkpoints_consistent": passed,
        "checkpoints": records,
        "continuous_board_stationarity_proved": False,
        "scope": "start/mid/end visual evidence with fixed head/body; operator maintained fixed board between checkpoints",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
