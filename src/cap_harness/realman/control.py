"""Bounded RM75 controller-native motion; no FK assumptions or implicit enabling."""

from __future__ import annotations

import copy
import math
import threading
import time

ARMS = {"left": 0, "right": 1}
# Different sources have different rates. The arm controller is queried at each
# cycle; the ~2 Hz whole-body stream has a finite period+jitter budget.
ROS_CONTEXT_DEADLINE_S = 0.75
CONTROLLER_DEADLINE_S = 0.2
FEEDBACK_POLL_S = 0.05
READ_COMMANDS = frozenset(
    {"get_current_arm_state", "get_current_tool_frame", "get_current_work_frame",
     "get_joint_min_pos", "get_joint_max_pos",
     "get_joint_drive_min_pos", "get_joint_drive_max_pos"}
)


def vector(value, count, *, integers=False):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"expected {count} values")
    if any(
        type(x) not in ((int,) if integers else (int, float)) or not math.isfinite(x) for x in value
    ):
        raise ValueError("non-finite, boolean or nonnumeric value")
    return list(value)


def arm_name(arm):
    if arm not in ARMS:
        raise ValueError("arm must be left or right")
    return arm


def decode_arm(response):
    if response.get("command") != "get_current_arm_state":
        raise ValueError("unexpected controller response")
    state = response["arm_state"]
    errors = state.get("err")
    if (
        not isinstance(errors, list)
        or not errors
        or any(type(x) is not int or x != 0 for x in errors)
    ):
        raise ValueError(f"controller errors: {errors}")
    return {
        "joint_deg": [x / 1000 for x in vector(state["joint"], 7, integers=True)],
        "pose_raw": vector(state["pose"], 6, integers=True),
        "raw": response,
    }


def state_joints(snapshot, arm, *, now=None, max_age_s=0.5):
    now = time.monotonic() if now is None else now
    age = now - snapshot["received_monotonic_s"]
    if not 0 <= age <= max_age_s:
        raise ValueError("stale ROS state")
    state = snapshot["state"]["arm_" + arm_name(arm)]
    if state.get("sys_err") != 0 or any(x != 0 for x in state.get("joint_err", [1])):
        raise ValueError("ROS arm error")
    if vector(state["joint_en"], 7, integers=True) != [1] * 7:
        raise ValueError("arm not enabled; enabling is never automatic")
    vector(state["joint_err"], 7, integers=True)
    return [x / 1000 for x in vector(state["position"], 7, integers=True)]


def max_delta(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("inconsistent joint vector")
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


class ArmControl:
    """One bounded command at a time, measured completion, latched failures.

    Transport must offer snapshot(), next_state(after, timeout), query(arm,payload),
    and audit(event). It is local to the robot, so WAN latency is not the servo loop.
    An envelope is an externally reviewed commissioning constraint, NOT a collision planner.
    """

    def __init__(self, transport, policy):
        self.transport = transport
        self.policy = copy.deepcopy(policy)
        self.lock = threading.Lock()
        self.dispatch_lock = threading.Lock()
        self.cancel = threading.Event()
        self.fault: str | None = None
        self.active_arm: str | None = None

    def blockers(self, arm):
        arm_name(arm)
        reasons = []
        for field in (
            "motion_enabled",
            "allow_stop",
            "exclusive_control_verified",
            "stop_verified",
        ):
            if self.policy.get(field) is not True:
                reasons.append(field)
        cfg = self.policy.get("arms", {}).get(arm, {})
        for field in (
            "controller_limits_deg",
            "reviewed_envelope_deg",
            "workspace_review",
            "limits_evidence",
        ):
            if not cfg.get(field):
                reasons.append(f"{arm}.{field}")
        if self.fault:
            reasons.append("fault_latched:" + self.fault)
        return reasons

    def read_arm(self, arm, *, timeout_s=None):
        arm_name(arm)
        started = time.monotonic()
        kwargs = {} if timeout_s is None else {"timeout_s": timeout_s}
        result = decode_arm(
            self.transport.query(arm, {"command": "get_current_arm_state"}, **kwargs)
        )
        if timeout_s is not None and time.monotonic() - started > timeout_s:
            raise TimeoutError("controller feedback deadline exceeded")
        return result

    def read_joint_limits(self, arm):
        """Read soft/drive limits; never install them or enable motion."""
        arm_name(arm)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("motion/capture owns the session")
        try:
            raw, values = {}, {}
            for name, command, key in (
                ("soft_min", "get_joint_min_pos", "min_pos"),
                ("soft_max", "get_joint_max_pos", "max_pos"),
                ("drive_min", "get_joint_drive_min_pos", "min_pos"),
                ("drive_max", "get_joint_drive_max_pos", "max_pos"),
            ):
                reply = self.transport.query(arm, {"command": command})
                # Firmware generations use either command or state as the tag.
                if not (reply.get("command") == command or
                        reply.get("state") == command.removeprefix("get_")):
                    raise ValueError("unexpected joint-limit response")
                if "command" in reply and reply["command"] != command:
                    raise ValueError("contradictory joint-limit response")
                if "state" in reply and reply["state"] != command.removeprefix("get_"):
                    raise ValueError("contradictory joint-limit state")
                raw[name] = reply
                values[name] = [q / 1000 for q in vector(reply[key], 7, integers=True)]
            intervals = []
            for i in range(7):
                lo = max(values["soft_min"][i], values["drive_min"][i])
                hi = min(values["soft_max"][i], values["drive_max"][i])
                if not lo < hi:
                    raise ValueError("empty or reversed joint-limit interval")
                intervals.append([lo, hi])
            return {"arm": arm, "units": "controller_degrees", "raw": raw,
                    "limits_deg": values, "effective_limits_deg": intervals,
                    "motion_policy_updated": False, "qualifies_motion": False}
        finally:
            self.lock.release()

    def check_bounds(self, arm, values):
        cfg = self.policy.get("arms", {}).get(arm, {})
        for key in ("controller_limits_deg", "reviewed_envelope_deg"):
            if cfg.get(key) is not None:
                bounds = cfg[key]
                if not isinstance(bounds, list) or len(bounds) != 7:
                    raise ValueError("seven joint intervals required")
                for q, interval in zip(values, bounds, strict=True):
                    lo, hi = vector(interval, 2)
                    if not lo < hi or not lo <= q <= hi:
                        raise ValueError(f"outside {key}")

    def preview(self, arm, target_deg, expected_start_deg, speed_percent=1):
        arm_name(arm)
        target = vector(target_deg, 7)
        expected = vector(expected_start_deg, 7)
        if type(speed_percent) is not int or speed_percent != 1:
            raise ValueError("commissioning supports controller speed percentage 1 only")
        target = [round(x * 1000) / 1000 for x in target]
        current = self.read_arm(arm)["joint_deg"]
        snapshot = self.transport.snapshot()
        measured = state_joints(snapshot, arm)
        if max_delta(current, measured) > 0.05 or max_delta(current, expected) > 0.05:
            raise ValueError("start state changed or controller/ROS feedback disagree")
        settled = self.transport.next_state(
            snapshot["received_monotonic_s"], timeout=ROS_CONTEXT_DEADLINE_S
        )
        if max_delta(measured, state_joints(settled, arm)) > 0.02:
            raise ValueError("arm is not stationary before motion")
        step = self.policy.get("max_step_deg", 0.5)
        if type(step) not in (int, float) or not math.isfinite(step) or not 0 < step <= 2:
            raise ValueError("invalid maximum step; commissioning ceiling is 2 degrees")
        if max_delta(current, target) > step:
            raise ValueError("target exceeds bounded commissioning step")
        cfg = self.policy.get("arms", {}).get(arm, {})
        self.check_bounds(arm, current)
        self.check_bounds(arm, target)
        if cfg.get("allowed_joint_indices") is not None:
            allowed = cfg["allowed_joint_indices"]
            if not isinstance(allowed, list) or any(
                type(i) is not int or i not in range(7) for i in allowed
            ):
                raise ValueError("invalid allowed joint indices")
            if any(
                abs(q1 - q0) > 0.0005
                for i, (q0, q1) in enumerate(zip(expected, target, strict=True))
                if i not in allowed
            ):
                raise ValueError("requested change to an unreviewed joint")
        if cfg.get("reviewed_context") is not None:
            self.check_context({"state": cfg["reviewed_context"]}, settled, arm)
        return {
            "arm": arm,
            "current_deg": current,
            "target_deg": target,
            "motion_blockers": self.blockers(arm),
            "collision_checked": False,
            "payload": {
                "command": "movej",
                "joint": [round(x * 1000) for x in target],
                "v": speed_percent,
                "r": 0,
                "trajectory_connect": 0,
            },
        }

    def stop(self, arm):
        arm_name(arm)
        if self.policy.get("allow_stop") is not True:
            raise ValueError("stop disabled in observation-only policy")
        self.cancel.set()
        self.fault = "stop_requested; restart and revalidate before further motion"
        with self.dispatch_lock:
            response = self.transport.query(arm, {"command": "set_arm_stop"})
        if response.get("command") != "set_arm_stop" or response.get("arm_stop") is not True:
            raise RuntimeError("stop not acknowledged; physical stop may be required")
        before = self.transport.next_state(time.monotonic(), timeout=1)
        q0 = state_joints(before, arm)
        after = self.transport.next_state(before["received_monotonic_s"], timeout=1)
        q1 = state_joints(after, arm)
        # An acknowledgement alone is never called a verified stop.
        result = {
            "arm": arm,
            "stop_acknowledged": True,
            "stationary_between_samples": max_delta(q0, q1) <= 0.02,
            "sample_span_s": after["received_monotonic_s"] - before["received_monotonic_s"],
            "joint_deg": q1,
            "hardware_estop_verified": False,
        }
        self.transport.audit({"event": "stop", **result})
        return result

    @staticmethod
    def check_context(before, after, moving_arm):
        for key in ("head", "body", "arm_" + ("right" if moving_arm == "left" else "left")):
            old, new = before["state"][key], after["state"][key]
            if new.get("sys_err") != 0 or any(x != 0 for x in new.get("joint_err", [1])):
                raise ValueError(f"context error: {key}")
            size = {"head": 2, "body": 5}.get(key, 7)
            old_q = vector(old["position"], size, integers=True)
            new_q = vector(new["position"], size, integers=True)
            if max_delta(old_q, new_q) > 100:
                raise ValueError(f"fixed body changed: {key}")
        chassis = after["state"]["chassis"]
        velocity = vector([chassis["linear"]["x"], chassis["angular"]["z"]], 2)
        if any(abs(x) > 0.001 for x in velocity):
            raise ValueError("chassis motion detected")

    def move(self, arm, target_deg, expected_start_deg, speed_percent=1):
        arm_name(arm)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("another motion/capture owns the session")
        sent = False
        try:
            reasons = self.blockers(arm)
            if reasons:
                raise ValueError("motion unavailable: " + ", ".join(reasons))
            self.cancel.clear()
            plan = self.preview(arm, target_deg, expected_start_deg, speed_percent)
            before = self.transport.snapshot()
            state_joints(before, arm)
            self.check_context(before, before, arm)
            self.transport.audit({"event": "motion_intent", **plan})
            with self.dispatch_lock:
                # A concurrent Stop during preflight must not be followed by a new MoveJ.
                if self.cancel.is_set() or self.fault:
                    raise RuntimeError("stopped during preflight; target not dispatched")
                self.active_arm = arm
                # Set before transmission: timeout is an unknown outcome, not a safe retry.
                sent = True
                reply = self.transport.query(arm, plan["payload"])
            if reply.get("command") != "movej" or reply.get("receive_state") is not True:
                raise RuntimeError("motion receipt not acknowledged")
            deadline = time.monotonic() + 15
            last = before
            stable_since = None
            ros_target_samples = 0
            feedback_samples = 0
            while time.monotonic() < deadline:
                if self.cancel.is_set():
                    raise RuntimeError("cancelled")
                # A short poll timeout means no NEW whole-body sample this cycle;
                # it is not permission to use a sample beyond its age deadline.
                try:
                    latest = self.transport.next_state(
                        last["received_monotonic_s"], timeout=FEEDBACK_POLL_S
                    )
                except TimeoutError:
                    latest = last
                measured = state_joints(latest, arm, max_age_s=ROS_CONTEXT_DEADLINE_S)
                self.check_context(before, latest, arm)
                started = time.monotonic()
                confirmed = self.read_arm(arm, timeout_s=CONTROLLER_DEADLINE_S)
                state_joints(latest, arm, max_age_s=ROS_CONTEXT_DEADLINE_S)
                self.check_bounds(arm, confirmed["joint_deg"])
                controller_ok = max_delta(confirmed["joint_deg"], plan["target_deg"]) <= 0.05
                ros_ok = max_delta(measured, plan["target_deg"]) <= 0.05
                if latest["received_monotonic_s"] > last["received_monotonic_s"]:
                    ros_target_samples = ros_target_samples + 1 if ros_ok else 0
                feedback_samples += 1
                self.transport.audit(
                    {
                        "event": "motion_feedback",
                        "arm": arm,
                        "joint_deg": confirmed["joint_deg"],
                        "ros_joint_deg": measured,
                        "ros_received_monotonic_s": latest["received_monotonic_s"],
                        "ros_age_s": time.monotonic() - latest["received_monotonic_s"],
                        "controller_query_s": time.monotonic() - started,
                    }
                )
                if controller_ok:
                    stable_since = time.monotonic() if stable_since is None else stable_since
                    if (
                        time.monotonic() - stable_since >= 0.3
                        and ros_ok
                        and ros_target_samples >= 2
                    ):
                        if self.cancel.is_set():
                            raise RuntimeError("cancelled during endpoint verification")
                        result = {
                            "status": "measured_endpoint_reached",
                            "arm": arm,
                            "joint_deg": confirmed["joint_deg"],
                            "target_deg": plan["target_deg"],
                            "max_error_deg": max_delta(confirmed["joint_deg"], plan["target_deg"]),
                            "feedback_samples": feedback_samples,
                            "fresh_ros_target_samples": ros_target_samples,
                            "object_task_success": None,
                        }
                        self.transport.audit({"event": "motion_complete", **result})
                        return result
                else:
                    stable_since = None
                last = latest
            raise TimeoutError("measured endpoint timeout")
        except Exception as exc:
            if sent:
                self.fault = str(exc)
                try:
                    stopped = self.stop(arm)
                except Exception as stop_exc:
                    stopped = {"stop_acknowledged": False, "error": str(stop_exc)}
                self.transport.audit({"event": "motion_failed", "error": str(exc), "stop": stopped})
            raise
        finally:
            self.active_arm = None
            self.lock.release()
