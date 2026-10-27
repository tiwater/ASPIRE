"""Offline reader for subscription-only ROS captures; respects image row strides."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .telemetry import decode_state


def image_array(folder: Path, metadata: dict) -> np.ndarray:
    enc = metadata["encoding"]
    formats = {
        "rgb8": ("u1", 3),
        "bgr8": ("u1", 3),
        "mono8": ("u1", 1),
        "16UC1": ("u2", 1),
        "32FC1": ("f4", 1),
    }
    if enc not in formats:
        raise ValueError(f"unsupported ROS image encoding: {enc}")
    dtype, channels = formats[enc]
    dtype = np.dtype((">" if metadata["is_bigendian"] else "<") + dtype)
    h, w, stride = metadata["height"], metadata["width"], metadata["step"]
    row_bytes = w * channels * dtype.itemsize
    if min(h, w) <= 0 or stride < row_bytes:
        raise ValueError("invalid image dimensions or stride")
    path = (folder / metadata["file"]).resolve()
    if path.parent != folder.resolve():
        raise ValueError("capture image path must stay in capture directory")
    data = path.read_bytes()
    if len(data) != h * stride:
        raise ValueError("image byte length does not match ROS Image metadata")
    packed = np.frombuffer(data, dtype=np.uint8).reshape(h, stride)[:, :row_bytes].copy()
    array = packed.view(dtype).reshape((h, w, channels) if channels > 1 else (h, w))
    return array[..., ::-1].copy() if enc == "bgr8" else array.copy()


def summarize(folder: Path, *, previews=False) -> dict:
    raw = json.loads((folder / "capture.json").read_text())
    images = {}
    for key, meta in raw["images"].items():
        a = image_array(folder, meta)
        stamp = meta["header"]["stamp"]
        sensor_s = stamp["sec"] + stamp["nanosec"] * 1e-9
        item = {
            "shape": list(a.shape),
            "encoding": meta["encoding"],
            "frame": meta["header"]["frame_id"],
            "sensor_timestamp_s": sensor_s,
            "receipt_minus_sensor_s": meta["received_unix_s"] - sensor_s,
            "has_camera_info": key in raw["camera_info"],
        }
        if a.ndim == 2:
            item["nonzero_fraction"] = float(np.mean(np.isfinite(a) & (a > 0)))
            item["depth_scale_to_m"] = None
            item["note"] = (
                "16UC1 is a storage encoding, not proof of millimetres. Confirm driver depth scale."
            )
        if previews and a.ndim == 3:
            from PIL import Image

            filename = key.strip("/").replace("/", "_") + ".png"
            Image.fromarray(a.astype(np.uint8)).save(folder / filename)
            item["preview"] = filename
        images[key] = item
    telemetry = (
        decode_state(
            raw["states"][-1]["state"], received_at_s=raw["states"][-1]["received_unix_s"]
        ).as_dict()
        if raw["states"]
        else None
    )
    summary = {
        "mode": raw["mode"],
        "images": images,
        "state": telemetry,
        "tf_children": sorted(raw["tf"]),
        "missing_images": raw["missing_images"],
        "cap_observation_ready": False,
        "pending": [
            "synchronized frames and states",
            "verified depth unit",
            "depth-to-color registration",
            "camera-to-body/arm transforms",
            "controller-to-URDF mapping",
        ],
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
