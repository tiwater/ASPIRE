"""Limit queries must remain read-only and fail closed on invalid replies."""
import copy
import threading

import pytest

from cap_harness.realman.control import READ_COMMANDS, ArmControl


class LimitsTransport:
    def __init__(self, mutate=None, state_tag=False):
        self.calls = []
        self.mutate = mutate
        self.state_tag = state_tag

    def query(self, arm, payload):
        cmd = payload["command"]
        self.calls.append((arm, payload.copy()))
        assert cmd in READ_COMMANDS and cmd.startswith("get_")
        lower = "min" in cmd
        n = 180000 if "drive" in cmd else 170000
        reply = {"min_pos" if lower else "max_pos": [(-n if lower else n)] * 7}
        reply["state" if self.state_tag else "command"] = (
            cmd.removeprefix("get_") if self.state_tag else cmd
        )
        if self.mutate:
            self.mutate(reply)
        return reply


@pytest.mark.parametrize("state_tag", [False, True])
def test_queries_intersect_bounds_without_changing_policy(state_tag):
    policy = {"motion_enabled": False}
    original = copy.deepcopy(policy)
    transport = LimitsTransport(state_tag=state_tag)
    control = ArmControl(transport, policy)
    result = control.read_joint_limits("right")
    assert result["effective_limits_deg"] == [[-170, 170]] * 7
    assert len(transport.calls) == 4
    assert all(a == "right" and len(p) == 1 for a, p in transport.calls)
    assert policy == original and not result["qualifies_motion"]


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(command="set_joint_min_pos"),
    lambda r: r.update(state="wrong_response"),
    lambda r: r.update(min_pos=[True] * 7),
    lambda r: r.update(min_pos=[0] * 6),
    lambda r: r.update(min_pos=[200000] * 7),
])
def test_invalid_limits_rejected_and_lock_released(mutation):
    control = ArmControl(LimitsTransport(mutation), {"motion_enabled": False})
    with pytest.raises((ValueError, KeyError)):
        control.read_joint_limits("right")
    assert control.lock.acquire(blocking=False)
    control.lock.release()


def test_limit_query_cannot_overlap_action():
    transport = LimitsTransport()
    control = ArmControl(transport, {"motion_enabled": False})
    control.lock = threading.Lock()
    control.lock.acquire()
    with pytest.raises(RuntimeError):
        control.read_joint_limits("right")
    assert not transport.calls
    control.lock.release()
