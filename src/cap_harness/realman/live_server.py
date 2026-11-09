"""RealMan native MCP entrypoint. Startup never homes, enables, clears errors or moves."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import fcntl
import json
from pathlib import Path
import time
from typing import Literal
import uuid

from pydantic import StrictFloat, StrictInt

from .control import (
    CONTROLLER_DEADLINE_S,
    FEEDBACK_POLL_S,
    ROS_CONTEXT_DEADLINE_S,
    ArmControl,
    arm_name,
    max_delta,
    state_joints,
)
from .handeye_capture import capture_pair
from .rgbd_capture import capture_rgbd
from .ros_transport import CAMERAS, RosTransport, frame_png


def build_server(control, *, shared=False, port=8232):
    from mcp.server.fastmcp import FastMCP, Image

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield {}
        finally:
            # Shared HTTP observers must not stop a different client's operation
            # merely by disconnecting. Request cancellation and process shutdown
            # still retain their existing stop paths.
            if not shared and control.active_arm is not None:
                await asyncio.to_thread(control.stop, control.active_arm)

    server = FastMCP(
        "ASPIRE RealMan",
        lifespan=lifespan,
        host="127.0.0.1",
        port=port,
        instructions="RealBot-01 with dual RM75. Controller-native seven-joint degrees. No Home, Cartesian poses, gripper or mobile-base motion exposed. Inspect capabilities before motion. Capture timing checks are not a hand-eye calibration certificate.",
    )

    @server.tool()
    def cap_get_capabilities() -> dict:
        """Return active motion gates and the robot-specific command contract."""
        return {
            "robot": "RealBot-01 / dual RM75",
            "arms": ["left", "right"],
            "joint_count": 7,
            "joint_units": "controller_degrees",
            "cameras": list(CAMERAS),
            "rgbd_capture_available": True,
            "motion_blockers": {a: control.blockers(a) for a in ("left", "right")},
            "stop_enabled": control.policy.get("allow_stop") is True,
            "collision_planner_available": False,
            "handeye_qualified": False,
            "transport": "ROS2 StringCmd /robot/command",
            "startup_motion": False,
            "feedback_supervision": {
                "controller_reply_deadline_s": CONTROLLER_DEADLINE_S,
                "ros_context_age_deadline_s": ROS_CONTEXT_DEADLINE_S,
                "poll_s": FEEDBACK_POLL_S,
                "endpoint_requires_new_ros_samples": 2,
            },
        }

    @server.tool()
    async def cap_probe_feedback(arm: Literal["left", "right"] = "left") -> dict:
        """Read-only 50-sample controller/ROS timing probe while stationary. Archives evidence; never enables motion."""

        def probe():
            if not control.lock.acquire(blocking=False):
                raise RuntimeError("motion/capture owns the session")
            samples, stamps = [], []
            try:
                initial = control.transport.next_state(time.monotonic(), timeout=2)
                start_q = state_joints(initial, arm)
                last = initial
                stamps.append(initial["received_monotonic_s"])
                for _ in range(50):
                    try:
                        last = control.transport.next_state(
                            last["received_monotonic_s"], timeout=FEEDBACK_POLL_S
                        )
                    except TimeoutError:
                        pass
                    state_joints(last, arm, max_age_s=ROS_CONTEXT_DEADLINE_S)
                    control.check_context(initial, last, arm)
                    started = time.monotonic()
                    result = control.read_arm(arm, timeout_s=CONTROLLER_DEADLINE_S)
                    elapsed = time.monotonic() - started
                    state_joints(last, arm, max_age_s=ROS_CONTEXT_DEADLINE_S)
                    if max_delta(start_q, result["joint_deg"]) > 0.02:
                        raise ValueError("arm moved during read-only timing probe")
                    if last["received_monotonic_s"] != stamps[-1]:
                        stamps.append(last["received_monotonic_s"])
                    samples.append(
                        {
                            "controller_query_s": elapsed,
                            "ros_age_s": time.monotonic() - last["received_monotonic_s"],
                            "joint_deg": result["joint_deg"],
                        }
                    )
                intervals = [b - a for a, b in zip(stamps, stamps[1:])]
                if len(intervals) < 3:
                    raise RuntimeError("insufficient fresh ROS messages")
                report = {
                    "arm": arm,
                    "samples": len(samples),
                    "controller_max_reply_s": max(s["controller_query_s"] for s in samples),
                    "ros_max_interval_s": max(intervals),
                    "ros_max_age_s": max(s["ros_age_s"] for s in samples),
                    "timing_probe_passed": True,
                    "qualifies_motion": False,
                }
                path = control.transport.artifact_root / (
                    "feedback_probe_" + uuid.uuid4().hex + ".json"
                )
                path.write_text(
                    json.dumps(
                        {**report, "observations": samples, "ros_intervals_s": intervals}, indent=2
                    )
                    + "\n"
                )
                return {**report, "path": str(path)}
            finally:
                control.lock.release()

        return await asyncio.to_thread(probe)

    @server.tool()
    async def cap_get_joint_limits(arm: Literal["left", "right"] = "right") -> dict:
        """Read seven soft/drive joint limits in controller degrees. Never changes policy or limits."""
        return await asyncio.to_thread(control.read_joint_limits, arm)

    @server.tool()
    async def cap_get_robot_state() -> dict:
        """Read fresh ROS state and both controller arm states; preserve raw pose units."""

        def read():
            return {
                "ros": control.transport.next_state(time.monotonic(), timeout=2),
                "controller": {a: control.read_arm(a) for a in ("left", "right")},
                "controller_pose_units": ["micrometre"] * 3 + ["milliradian"] * 3,
                "pose_frame_convention_verified": False,
            }

        return await asyncio.to_thread(read)

    @server.tool()
    async def cap_observe(camera: str = "left_arm_depth_cam") -> list:
        """Get one fresh RGB image and its CameraInfo; does not move or open another camera."""

        def read():
            frame = control.transport.frame(camera)
            return [
                json.dumps(
                    {
                        "camera": camera,
                        "source_age_s": frame["source_age_s"],
                        "camera_info": frame["camera_info"],
                    }
                ),
                Image(data=frame_png(frame), format="png"),
            ]

        return await asyncio.to_thread(read)

    @server.tool()
    async def cap_capture_rgbd(camera: Literal["head_depth_cam", "chest_depth_cam", "left_arm_depth_cam", "right_arm_depth_cam"] = "head_depth_cam") -> dict:
        """Archive fresh raw RGB/depth, both intrinsics and driver depth-to-color TF. No motion or world localization."""
        return await asyncio.to_thread(capture_rgbd, control, camera)

    @server.tool()
    async def cap_preview_joint_move(
        arm: Literal["left", "right"],
        target_deg: list[StrictFloat],
        expected_start_deg: list[StrictFloat],
        speed_percent: StrictInt = 1,
    ) -> dict:
        """Read-only check of seven controller targets. Reports blockers; no collision certificate."""
        return await asyncio.to_thread(
            control.preview, arm, target_deg, expected_start_deg, speed_percent
        )

    @server.tool()
    async def cap_move_to_joints(
        arm: Literal["left", "right"],
        target_deg: list[StrictFloat],
        expected_start_deg: list[StrictFloat],
        speed_percent: StrictInt = 1,
    ) -> list:
        """Bounded seven-joint move, only after commissioning gates. Wait for measured endpoint; return image."""
        try:
            result = await asyncio.to_thread(
                control.move, arm, target_deg, expected_start_deg, speed_percent
            )
        except asyncio.CancelledError:
            if control.active_arm == arm:
                await asyncio.shield(asyncio.to_thread(control.stop, arm))
            raise
        # Preserve terminal motion evidence if the optional post-motion camera
        # fails. Turning a completed move into a generic tool error invites an
        # unsafe replay. A new observation may be retried; this move may not.
        try:
            frame = await asyncio.to_thread(control.transport.frame, arm + "_arm_depth_cam")
            preview = Image(data=frame_png(frame), format="png")
        except (ValueError, TimeoutError, RuntimeError) as exc:
            control.transport.audit(
                {"event": "post_motion_observation_failed", "arm": arm, "error": str(exc)}
            )
            return [
                json.dumps(
                    {
                        **result,
                        "observation_available": False,
                        "observation_error": str(exc),
                        "next_required_tool": "cap_observe",
                        "replay_motion": False,
                    }
                )
            ]
        return [json.dumps({**result, "observation_available": True}), preview]

    @server.tool()
    async def cap_stop(arm: str) -> dict:
        """Stop one arm and inspect measured stationarity. Available only when stop is enabled."""
        arm_name(arm)
        return await asyncio.to_thread(control.stop, arm)

    @server.tool()
    async def cap_capture_handeye_sample(arm: str = "left") -> dict:
        """Read-only stationary pose/image bracket, archived on robot. Does not qualify hand-eye."""
        return await asyncio.to_thread(capture_pair, control, arm)

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--lock-file", required=True, type=Path)
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--port", type=int, default=8232)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("unknown control policy version")
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport = RosTransport(
            args.artifacts,
            allow_motion=policy.get("motion_enabled") is True,
            allow_stop=policy.get("allow_stop") is True,
        )
        control = ArmControl(transport, policy)
        try:
            transport.audit({"event": "server_start", "policy": policy, "startup_motion": False})
            build_server(control, shared=args.transport == "streamable-http", port=args.port).run(
                transport=args.transport
            )
        finally:
            try:
                if control.active_arm is not None:
                    control.stop(control.active_arm)
            finally:
                transport.close()


if __name__ == "__main__":
    main()
