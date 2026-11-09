"""Read-only RGB-D archives. No controller query, actuator command, or world-frame claim."""

from __future__ import annotations

import hashlib
import json
import time
import uuid

import numpy as np

from .ros_transport import frame_png, stamp_dict


def stamp_ns(message):
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def choose_pair(colors, depths, *, after, now, ros_now_ns):
    def fresh(frame):
        return (after < frame["received_monotonic_s"] <= now
                and now - frame["received_monotonic_s"] <= 0.5
                and -0.05 <= (ros_now_ns - stamp_ns(frame["message"])) / 1e9 <= 0.5)
    pairs = [(c, d) for c in colors if fresh(c) for d in depths if fresh(d)
             and abs(stamp_ns(c["message"]) - stamp_ns(d["message"])) <= 50_000_000]
    if not pairs:
        return None
    return min(pairs, key=lambda p: (abs(stamp_ns(p[0]["message"]) - stamp_ns(p[1]["message"])),
                                    -min(stamp_ns(p[0]["message"]), stamp_ns(p[1]["message"]))))


def validate_frame(frame):
    msg, info = frame["message"], frame["camera_info"]
    if (msg.width, msg.height, msg.header.frame_id) != (info["width"], info["height"], info["header"]["frame_id"]):
        raise ValueError("image/CameraInfo dimensions or frame disagree")
    s = info["header"]["stamp"]
    if abs(stamp_ns(msg) - (s["sec"] * 1_000_000_000 + s["nanosec"])) > 1_000_000_000:
        raise ValueError("CameraInfo timestamp does not match current stream")
    k = np.asarray(info["k"], dtype=float)
    if k.size != 9 or not np.isfinite(k).all() or k[0] <= 0 or k[4] <= 0:
        raise ValueError("invalid camera intrinsics")


def depth_metres(msg):
    # REP118: 16UC1 millimetres or 32FC1 metres, camera optical Z, never RGB-indexed.
    if msg.encoding not in ("16UC1", "32FC1"):
        raise ValueError("unsupported depth encoding")
    dtype = np.dtype((">" if msg.is_bigendian else "<") + ("u2" if msg.encoding == "16UC1" else "f4"))
    if msg.width <= 0 or msg.height <= 0 or msg.step < msg.width * dtype.itemsize or msg.step % dtype.itemsize:
        raise ValueError("invalid depth dimensions/stride")
    if len(msg.data) != msg.height * msg.step:
        raise ValueError("depth buffer length mismatch")
    data = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, msg.step // dtype.itemsize)
    result = data[:, :msg.width].astype(np.float32)
    if msg.encoding == "16UC1":
        result *= 0.001
    result[~np.isfinite(result) | (result <= 0)] = np.nan
    return result


def capture_rgbd(control, camera):
    if not control.lock.acquire(blocking=False):
        raise RuntimeError("motion/capture owns the session")
    try:
        transport = control.transport
        started = time.monotonic()
        before = transport.next_state(started, timeout=2)
        color, depth, registration = transport.rgbd_pair(camera, after=before["received_monotonic_s"])
        after = transport.next_state(max(color["received_monotonic_s"], depth["received_monotonic_s"]), timeout=2)
        if registration["parent_frame"] != color["message"].header.frame_id or registration["child_frame"] != depth["message"].header.frame_id:
            raise ValueError("depth-to-color transform direction mismatch")
        q = np.asarray(registration["quaternion_xyzw"], dtype=float)
        t = np.asarray(registration["translation_m"], dtype=float)
        if q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(t).all() or abs(np.linalg.norm(q) - 1) > 1e-4:
            raise ValueError("invalid depth-to-color transform")
        metres = depth_metres(depth["message"])
        valid = np.isfinite(metres)
        if not valid.any():
            raise ValueError("depth image has no positive finite measurements")
        directory = transport.artifact_root / ("rgbd_" + uuid.uuid4().hex)
        directory.mkdir()
        (directory / "color.png").write_bytes(frame_png(color))
        (directory / "depth.bin").write_bytes(bytes(depth["message"].data))
        np.save(directory / "depth_m.npy", metres, allow_pickle=False)
        metadata = {}
        for name, frame in [("color", color), ("depth", depth)]:
            msg = frame["message"]
            metadata[name] = {"header": stamp_dict(msg.header), "encoding": msg.encoding,
                              "width": msg.width, "height": msg.height, "step": msg.step,
                              "is_bigendian": int(msg.is_bigendian), "camera_info": frame["camera_info"],
                              "received_monotonic_s": frame["received_monotonic_s"]}
        deltas = {key: float(np.max(np.abs(np.array(after["state"][key]["position"]) - before["state"][key]["position"]))) / 1000
                  for key in ["arm_left", "arm_right", "head", "body"]}
        report = {"camera": camera, **metadata, "color_from_depth": registration,
                  "source_skew_s": abs(stamp_ns(color["message"]) - stamp_ns(depth["message"])) / 1e9,
                  "hardware_synchronized": False, "depth_units": "metres along depth optical Z",
                  "depth_registered_to_rgb": False,
                  "depth_valid_fraction": float(valid.mean()),
                  "depth_valid_percentiles_m": np.percentile(metres[valid], [5, 50, 95]).tolist(),
                  "state_before": before, "state_after": after,
                  "endpoint_joint_deltas_deg": deltas,
                  "whole_interval_stationarity_verified": False,
                  "base_or_world_transform_provided": False, "motion_commanded": False,
                  "limitations": ["Depth units follow REP118; device depth scale not independently calibrated",
                                  "Project through depth intrinsics and driver depth-to-color TF; do not index depth with RGB pixels",
                                  "Endpoint state agreement does not prove stationarity throughout exposure interval",
                                  "No calibrated target localization, collision clearance or grasp success claimed"],
                  "sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                             for name in ["color.png", "depth.bin", "depth_m.npy"]}}
        (directory / "sample.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return {"path": str(directory), **report}
    finally:
        control.lock.release()
