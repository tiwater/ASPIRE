"""Failure and gating tests use a fake plant; never access robot or ROS."""

import asyncio
import copy
import json
from pathlib import Path
import time

import pytest

from cap_harness.realman.control import ArmControl, decode_arm, state_joints, vector
from cap_harness.realman.handeye_capture import check_pair


def raw_state(q=None):
    state = {}
    for name, n in [("arm_left", 7), ("arm_right", 7), ("body", 5), ("head", 2)]:
        state[name] = {"position": [0] * n, "joint_en": [1] * n, "joint_err": [0] * n, "sys_err": 0}
    state["arm_left"]["position"] = q or [0] * 7
    state["chassis"] = {"linear": {"x": 0.0}, "angular": {"z": 0.0}}
    return state


def controller_response(q=None):
    return {
        "command": "get_current_arm_state",
        "arm_state": {"joint": q or [0] * 7, "pose": [1000, 2000, 3000, 0, 0, 0], "err": [0]},
    }


class Plant:
    def __init__(self, monkeypatch):
        self.now = 100.0
        monkeypatch.setattr(time, "monotonic", lambda: self.now)
        self.raw = raw_state()
        self.commands = []
        self.events = []
        self.behavior = "ok"
        self.sent = False
        self.snapshot_age = 0.0

    def snapshot(self):
        return {
            "received_monotonic_s": self.now - self.snapshot_age,
            "state": copy.deepcopy(self.raw),
        }

    def next_state(self, after, timeout):
        self.now = max(self.now, after) + 0.1
        if self.sent and self.behavior == "state_timeout":
            raise TimeoutError("lost state")
        if self.sent and self.behavior == "body_changed":
            self.raw["body"]["position"][0] = 200
        return self.snapshot()

    def query(self, arm, payload, *, timeout_s=2.0):
        command = payload["command"]
        self.commands.append((arm, copy.deepcopy(payload)))
        if command == "get_current_arm_state":
            return controller_response(self.raw["arm_" + arm]["position"])
        if command == "set_arm_stop":
            return {"command": command, "arm_stop": self.behavior != "stop_rejected"}
        assert command == "movej"
        self.sent = True
        if self.behavior == "ack_lost":
            raise TimeoutError("unknown transmission outcome")
        if self.behavior != "never_arrives":
            self.raw["arm_" + arm]["position"] = payload["joint"]
        return {"command": command, "receive_state": self.behavior != "rejected"}

    def audit(self, value):
        self.events.append(value)


def policy():
    result = json.loads(Path("configs/realman/control_policy.json").read_text())
    result.update(
        motion_enabled=True, allow_stop=True, exclusive_control_verified=True, stop_verified=True
    )
    for arm in result["arms"].values():
        arm.update(
            controller_limits_deg=[[-180, 180]] * 7,
            reviewed_envelope_deg=[[-1, 1]] * 7,
            workspace_review="fake test scene",
            limits_evidence="fake test limits",
        )
    return result


@pytest.mark.parametrize(
    "value", [[True] * 7, [float("nan")] * 7, [float("inf")] * 7, [0] * 6, "0" * 7]
)
def test_reject_bad_vectors(value):
    with pytest.raises(ValueError):
        vector(value, 7)


def test_shipping_policy_cannot_send_motion(monkeypatch):
    plant = Plant(monkeypatch)
    control = ArmControl(plant, json.loads(Path("configs/realman/control_policy.json").read_text()))
    with pytest.raises(ValueError, match="motion unavailable"):
        control.move("left", [0.2] + [0] * 6, [0] * 7)
    with pytest.raises(ValueError, match="stop disabled"):
        control.stop("left")
    assert plant.commands == []


@pytest.mark.parametrize(
    "gate", ["motion_enabled", "allow_stop", "exclusive_control_verified", "stop_verified"]
)
def test_each_gate_is_required(monkeypatch, gate):
    plant = Plant(monkeypatch)
    cfg = policy()
    cfg[gate] = False
    with pytest.raises(ValueError, match="motion unavailable"):
        ArmControl(plant, cfg).move("left", [0.2] + [0] * 6, [0] * 7)
    assert not plant.commands


@pytest.mark.parametrize(
    "target,start,speed",
    [
        ([0.6] + [0] * 6, [0] * 7, 1),
        ([0.2] + [0] * 6, [0.1] + [0] * 6, 1),
        ([0.2] + [0] * 6, [0] * 7, 2),
    ],
)
def test_invalid_plan_sends_no_motion(monkeypatch, target, start, speed):
    plant = Plant(monkeypatch)
    with pytest.raises(ValueError):
        ArmControl(plant, policy()).move("left", target, start, speed)
    assert all(c[1]["command"] == "get_current_arm_state" for c in plant.commands)


def test_measured_completion_and_exact_protocol(monkeypatch):
    plant = Plant(monkeypatch)
    result = ArmControl(plant, policy()).move("left", [0.2] + [0] * 6, [0] * 7)
    moves = [p for a, p in plant.commands if p["command"] == "movej"]
    assert moves == [
        {
            "command": "movej",
            "joint": [200, 0, 0, 0, 0, 0, 0],
            "v": 1,
            "r": 0,
            "trajectory_connect": 0,
        }
    ]
    assert result["status"] == "measured_endpoint_reached"
    assert result["max_error_deg"] == 0
    assert plant.now >= 100.4


@pytest.mark.parametrize(
    "failure", ["ack_lost", "rejected", "never_arrives", "state_timeout", "body_changed"]
)
def test_failure_stops_latches_and_never_replays(monkeypatch, failure):
    plant = Plant(monkeypatch)
    plant.behavior = failure
    control = ArmControl(plant, policy())
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        control.move("left", [0.2] + [0] * 6, [0] * 7)
    assert sum(p["command"] == "movej" for a, p in plant.commands) == 1
    assert any(a == "left" and p["command"] == "set_arm_stop" for a, p in plant.commands)
    assert control.fault
    with pytest.raises(ValueError, match="motion unavailable"):
        control.move("left", [0.2] + [0] * 6, [0] * 7)
    assert sum(p["command"] == "movej" for a, p in plant.commands) == 1


def test_stale_state_and_wrong_joint_count(monkeypatch):
    plant = Plant(monkeypatch)
    plant.snapshot_age = 1
    with pytest.raises(ValueError, match="stale"):
        ArmControl(plant, policy()).preview("left", [0] * 7, [0] * 7)
    reply = controller_response()
    reply["arm_state"]["joint"] = [0] * 6
    with pytest.raises(ValueError):
        decode_arm(reply)


def test_joint_envelope_is_enforced(monkeypatch):
    plant = Plant(monkeypatch)
    cfg = policy()
    cfg["arms"]["left"]["reviewed_envelope_deg"] = [[-0.1, 0.1]] * 7
    with pytest.raises(ValueError, match="outside"):
        ArmControl(plant, cfg).move("left", [0.2] + [0] * 6, [0] * 7)
    assert not plant.sent


def test_concurrent_owner_rejected(monkeypatch):
    plant = Plant(monkeypatch)
    control = ArmControl(plant, policy())
    with control.lock, pytest.raises(RuntimeError, match="owns"):
        control.move("left", [0] * 7, [0] * 7)
    assert not plant.commands


def test_pair_requires_stationarity_and_never_qualifies(monkeypatch):
    Plant(monkeypatch)
    states = [{"received_monotonic_s": 100 + i * 0.1, "state": raw_state()} for i in range(6)]
    result = check_pair(controller_response(), controller_response(), states, "left")
    assert result["timing_stationarity_passed"] is True
    assert result["hardware_synchronized"] is False
    assert result["handeye_qualified"] is False
    states[2]["state"]["arm_left"]["position"][0] = 100
    with pytest.raises(ValueError, match="disagree or moved"):
        check_pair(controller_response(), controller_response(), states, "left")


def test_controller_error_and_disabled_joint(monkeypatch):
    plant = Plant(monkeypatch)
    reply = controller_response()
    reply["arm_state"]["err"] = [1]
    with pytest.raises(ValueError, match="controller errors"):
        decode_arm(reply)
    plant.raw["arm_left"]["joint_en"][1] = 0
    with pytest.raises(ValueError, match="not enabled"):
        state_joints(plant.snapshot(), "left")


def test_mcp_schema_and_default_gates(monkeypatch):
    pytest.importorskip("mcp")
    from cap_harness.realman.live_server import build_server

    plant = Plant(monkeypatch)
    server = build_server(
        ArmControl(plant, json.loads(Path("configs/realman/control_policy.json").read_text()))
    )
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert "cap_move_to_joints" in names and "cap_capture_handeye_sample" in names
    assert "cap_program_go_home" not in names
    motion = next(t for t in tools if t.name == "cap_move_to_joints")
    assert "target_deg" in motion.inputSchema["properties"]
    assert motion.inputSchema["properties"]["arm"]["enum"] == ["left", "right"]
    assert not plant.commands


def test_stop_rejection_is_not_reported_as_success(monkeypatch):
    plant = Plant(monkeypatch)
    plant.behavior = "stop_rejected"
    control = ArmControl(plant, policy())
    with pytest.raises(RuntimeError, match="stop not acknowledged"):
        control.stop("left")
    assert control.fault


def test_pair_rejects_motion_in_other_body(monkeypatch):
    Plant(monkeypatch)
    states = [{"received_monotonic_s": 100 + i * 0.1, "state": raw_state()} for i in range(6)]
    states[3]["state"]["head"]["position"][1] = 150
    with pytest.raises(ValueError, match="fixed body changed"):
        check_pair(controller_response(), controller_response(), states, "left")


def test_mcp_rejects_boolean_numeric_input(monkeypatch):
    pytest.importorskip("mcp")
    from cap_harness.realman.live_server import build_server

    plant = Plant(monkeypatch)
    server = build_server(ArmControl(plant, policy()))
    with pytest.raises(Exception, match="valid number"):
        asyncio.run(
            server.call_tool(
                "cap_preview_joint_move",
                {"arm": "left", "target_deg": [True] + [0.0] * 6, "expected_start_deg": [0.0] * 7},
            )
        )
    assert not plant.commands


def test_disk_failure_cannot_suppress_stop_audit_path():
    from cap_harness.realman.ros_transport import RosTransport

    transport = object.__new__(RosTransport)

    def disk_full(event):
        raise OSError("disk full")

    transport.audit = disk_full
    with pytest.raises(OSError, match="disk full"):
        transport._query_audit("movej", {})
    transport._query_audit("set_arm_stop", {})


def test_stationary_capture_waits_for_three_low_rate_messages(monkeypatch):
    from cap_harness.realman.handeye_capture import stationary_window

    plant = Plant(monkeypatch)
    history = []

    def next_sample(after, timeout):
        assert timeout == 2
        plant.now = max(plant.now, after) + 0.6
        value = plant.snapshot()
        history.append(value)
        return value

    plant.next_state = next_sample
    plant.state_window = lambda start, end: [
        x for x in history if start <= x["received_monotonic_s"] <= end
    ]
    first = stationary_window(plant)
    assert len(history) == 3
    assert first == history[0]
    assert history[-1]["received_monotonic_s"] - first["received_monotonic_s"] >= 1.0


def test_capture_directory_failure_releases_owner(monkeypatch, tmp_path):
    from cap_harness.realman.handeye_capture import capture_pair

    plant = Plant(monkeypatch)
    plant.artifact_root = tmp_path / "missing-parent"
    control = ArmControl(plant, policy())
    with pytest.raises(FileNotFoundError):
        capture_pair(control, "left")
    assert not control.lock.locked()
    assert not plant.commands


def test_stop_during_preflight_cannot_be_followed_by_move(monkeypatch):
    plant = Plant(monkeypatch)
    control = ArmControl(plant, policy())
    original = plant.query
    intercepted = False

    def query(arm, payload):
        nonlocal intercepted
        if payload["command"] == "get_current_arm_state" and not intercepted:
            intercepted = True
            control.stop(arm)
        return original(arm, payload)

    plant.query = query
    with pytest.raises(RuntimeError, match="stopped during preflight"):
        control.move("left", [0.2] + [0] * 6, [0] * 7)
    assert any(p["command"] == "set_arm_stop" for a, p in plant.commands)
    assert not any(p["command"] == "movej" for a, p in plant.commands)


def test_shared_observer_disconnect_does_not_stop_another_operation():
    pytest.importorskip("mcp")
    from types import SimpleNamespace

    from cap_harness.realman.live_server import build_server

    stopped = []
    control = SimpleNamespace(active_arm="left", stop=lambda arm: stopped.append(arm))
    server = build_server(control, shared=True)

    async def observer_lifetime():
        async with server._mcp_server.lifespan(server._mcp_server):
            pass

    asyncio.run(observer_lifetime())
    assert not stopped
