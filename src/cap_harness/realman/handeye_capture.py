"""Stationary controller-pose brackets around a new wrist image; not hardware sync."""

from __future__ import annotations

import json
import time
import uuid

from .control import ArmControl, arm_name, decode_arm, max_delta, state_joints
from .ros_transport import frame_png, stamp_dict


def stationary_window(transport):
    """Wait for new low-rate telemetry during read-only, stationary acquisition."""
    first = transport.next_state(time.monotonic(), timeout=2)
    latest = first
    deadline = time.monotonic() + 5
    while True:
        samples = transport.state_window(
            first["received_monotonic_s"], latest["received_monotonic_s"]
        )
        if (
            len(samples) >= 3
            and latest["received_monotonic_s"] - first["received_monotonic_s"] >= 0.5
        ):
            return first
        if time.monotonic() > deadline:
            raise TimeoutError("stationarity window timeout")
        latest = transport.next_state(latest["received_monotonic_s"], timeout=2)


def check_pair(before, after, states, arm):
    q0, q1 = decode_arm(before), decode_arm(after)
    if max_delta(q0["joint_deg"], q1["joint_deg"]) > 0.02:
        raise ValueError("controller joints moved during exposure bracket")
    if max_delta(q0["pose_raw"][:3], q1["pose_raw"][:3]) > 300:
        raise ValueError("controller translation changed over 0.3 mm")
    if max_delta(q0["pose_raw"][3:], q1["pose_raw"][3:]) > 2:
        raise ValueError("controller orientation changed over 2 mrad")
    if (
        len(states) < 3
        or states[-1]["received_monotonic_s"] - states[0]["received_monotonic_s"] < 0.4
    ):
        raise ValueError("insufficient stationary ROS time span")
    for item in states:
        joints = state_joints(item, arm, now=item["received_monotonic_s"])
        if max_delta(q0["joint_deg"], joints) > 0.02:
            raise ValueError("ROS/controller joints disagree or moved")
        ArmControl.check_context(states[0], item, arm)
    return {
        "timing_stationarity_passed": True,
        "hardware_synchronized": False,
        "handeye_qualified": False,
        "pose_frame_convention_verified": False,
        "board_stationarity_verified": False,
        "note": "Stationary software bracket; retain raw poses and timestamps. Requires board detection, fixed-board evidence and pose convention verification before solving.",
    }


def capture_pair(control, arm):
    arm_name(arm)
    if not control.lock.acquire(blocking=False):
        raise RuntimeError("motion or another capture in progress")
    transport = control.transport
    folder = transport.artifact_root / ("handeye_" + arm + "_" + uuid.uuid4().hex)
    try:
        folder.mkdir()
    except OSError:
        control.lock.release()
        raise
    record = {
        "arm": arm,
        "hardware_synchronized": False,
        "handeye_qualified": False,
        "timing_stationarity_passed": False,
        "started_unix_s": time.time(),
    }
    try:
        # Read-only acquisition tolerates low-rate telemetry; motion guards stay strict.
        first = stationary_window(transport)
        state_joints(transport.snapshot(), arm)
        for name in ("tool", "work"):
            record[name + "_frame_before"] = transport.query(
                arm, {"command": f"get_current_{name}_frame"}
            )
        record["pose_before_request_monotonic_s"] = time.monotonic()
        record["pose_before"] = transport.query(arm, {"command": "get_current_arm_state"})
        record["pose_before_received_monotonic_s"] = time.monotonic()
        anchor_ns = transport.node.get_clock().now().nanoseconds
        record["pose_before_received_ros_ns"] = anchor_ns
        camera = arm + "_arm_depth_cam"
        after_receipt = record["pose_before_received_monotonic_s"]
        image_deadline = time.monotonic() + 1
        while True:
            remaining = image_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no image exposure after the first pose query")
            frame = transport.frame(camera, after=after_receipt, timeout=remaining)
            msg = frame["message"]
            source_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            if source_ns > anchor_ns:
                break
            after_receipt = frame["received_monotonic_s"]
        record["pose_after_request_monotonic_s"] = time.monotonic()
        record["pose_after_request_ros_ns"] = transport.node.get_clock().now().nanoseconds
        if source_ns > record["pose_after_request_ros_ns"]:
            raise ValueError("image timestamp is outside controller pose bracket")
        record["pose_after"] = transport.query(arm, {"command": "get_current_arm_state"})
        record["pose_after_received_monotonic_s"] = time.monotonic()
        span = record["pose_after_received_monotonic_s"] - record["pose_before_request_monotonic_s"]
        if span > 1:
            raise ValueError("pose bracket exceeds 1 second")
        latest = transport.next_state(record["pose_after_received_monotonic_s"], 2)
        record["states"] = transport.state_window(
            first["received_monotonic_s"], latest["received_monotonic_s"]
        )
        for name in ("tool", "work"):
            record[name + "_frame_after"] = transport.query(
                arm, {"command": f"get_current_{name}_frame"}
            )
            if record[name + "_frame_before"] != record[name + "_frame_after"]:
                raise ValueError(f"{name} frame configuration changed")
        record["image"] = {
            "header": stamp_dict(msg.header),
            "encoding": msg.encoding,
            "width": msg.width,
            "height": msg.height,
            "step": msg.step,
            "is_bigendian": msg.is_bigendian,
            "received_monotonic_s": frame["received_monotonic_s"],
            "received_unix_s": frame["received_unix_s"],
            "source_age_s_at_read": frame["source_age_s"],
            "file": "image.bin",
            "preview": "image.png",
        }
        record["camera_info"] = frame["camera_info"]
        record["bracket_span_s"] = span
        (folder / "image.bin").write_bytes(bytes(msg.data))
        (folder / "image.png").write_bytes(frame_png(frame))
        record.update(
            check_pair(record["pose_before"], record["pose_after"], record["states"], arm)
        )
    except Exception as exc:
        record["error"] = str(exc)
    finally:
        record["finished_unix_s"] = time.time()
        try:
            (folder / "sample.json").write_text(
                json.dumps(record, indent=2, allow_nan=False) + "\n"
            )
        finally:
            control.lock.release()
    transport.audit(
        {
            "event": "handeye_capture",
            "path": str(folder),
            "timing_stationarity_passed": record["timing_stationarity_passed"],
            "error": record.get("error"),
        }
    )
    return {"path": str(folder), **{k: v for k, v in record.items() if k not in ("states",)}}
