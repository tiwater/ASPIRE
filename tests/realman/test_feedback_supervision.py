"""Independent clocks for fast controller replies and jittered 2 Hz ROS status."""

import copy

import pytest
from test_live_control import Plant, controller_response, policy

from cap_harness.realman.control import ArmControl


class JitterPlant(Plant):
    def __init__(self, monkeypatch):
        super().__init__(monkeypatch)
        self.last_ros = self.now
        self.ros_raw = copy.deepcopy(self.raw)
        self.period = 0.504
        self.drop_ros = False
        self.slow_controller = False
        self.controller_wrong = False

    def snapshot(self):
        return {"received_monotonic_s": self.last_ros, "state": copy.deepcopy(self.ros_raw)}

    def next_state(self, after, timeout):
        next_ros = self.last_ros + self.period
        if next_ros > self.now + timeout or (self.sent and self.drop_ros):
            self.now += timeout
            raise TimeoutError("fresh ROS state timeout")
        self.now = max(self.now, next_ros)
        self.last_ros = self.now
        self.ros_raw = copy.deepcopy(self.raw)
        return self.snapshot()

    def query(self, arm, payload, *, timeout_s=2.0):
        self.now += 0.201 if self.sent and self.slow_controller else 0.008
        result = super().query(arm, payload, timeout_s=timeout_s)
        if self.sent and self.controller_wrong and payload["command"] == "get_current_arm_state":
            return controller_response()
        return result


def test_504ms_ros_period_completes_with_fast_feedback_and_two_new_ros_samples(monkeypatch):
    plant = JitterPlant(monkeypatch)
    result = ArmControl(plant, policy()).move("left", [0.2] + [0] * 6, [0] * 7)
    assert result["status"] == "measured_endpoint_reached"
    assert result["fresh_ros_target_samples"] >= 2
    assert result["feedback_samples"] >= 10
    assert not any(p["command"] == "set_arm_stop" for _, p in plant.commands)
    assert sum(p["command"] == "movej" for _, p in plant.commands) == 1


@pytest.mark.parametrize("fault", ["drop_ros", "slow_controller", "controller_wrong"])
def test_either_feedback_path_failure_stops_without_replay(monkeypatch, fault):
    plant = JitterPlant(monkeypatch)
    setattr(plant, fault, True)
    control = ArmControl(plant, policy())
    with pytest.raises((ValueError, TimeoutError)):
        control.move("left", [0.2] + [0] * 6, [0] * 7)
    assert sum(p["command"] == "movej" for _, p in plant.commands) == 1
    assert any(p["command"] == "set_arm_stop" for _, p in plant.commands)
    assert control.fault
    if fault != "controller_wrong":
        assert plant.now < 103.0


def test_current_feedback_outside_reviewed_envelope_stops(monkeypatch):
    plant = JitterPlant(monkeypatch)
    original = plant.query

    def query(arm, payload, **kwargs):
        result = original(arm, payload, **kwargs)
        if plant.sent and payload["command"] == "get_current_arm_state":
            return controller_response([400] + [0] * 6)
        return result

    plant.query = query
    cfg = policy()
    cfg["arms"]["left"]["reviewed_envelope_deg"] = [[-0.3, 0.3]] * 7
    with pytest.raises(ValueError, match="outside reviewed_envelope"):
        ArmControl(plant, cfg).move("left", [0.2] + [0] * 6, [0] * 7)
    assert any(p["command"] == "set_arm_stop" for _, p in plant.commands)


def test_reviewed_scene_context_checked_before_dispatch(monkeypatch):
    plant = Plant(monkeypatch)
    cfg = policy()
    cfg["arms"]["left"]["reviewed_context"] = copy.deepcopy(plant.raw)
    plant.raw["head"]["position"][0] = 200
    with pytest.raises(ValueError, match="fixed body changed"):
        ArmControl(plant, cfg).move("left", [0.2] + [0] * 6, [0] * 7)
    assert not plant.sent


def test_only_reviewed_joints_may_be_requested(monkeypatch):
    plant = Plant(monkeypatch)
    cfg = policy()
    cfg["arms"]["left"]["allowed_joint_indices"] = [6]
    with pytest.raises(ValueError, match="unreviewed joint"):
        ArmControl(plant, cfg).move("left", [0.2] + [0] * 6, [0] * 7)
    assert not plant.sent
