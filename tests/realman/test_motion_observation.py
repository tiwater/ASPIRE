"""A stale post-motion image must not erase measured completion or replay motion."""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")
from cap_harness.realman.live_server import build_server


def test_completed_move_survives_camera_failure():
    calls, events = [], []

    def move(*args):
        calls.append(args)
        return {"status": "measured_endpoint_reached", "joint_deg": [0.0] * 7}

    def frame(*args):
        raise ValueError("image source or receipt timestamp stale/inconsistent")

    control = SimpleNamespace(
        move=move, active_arm=None, transport=SimpleNamespace(frame=frame, audit=events.append)
    )
    server = build_server(control)
    result = asyncio.run(
        server.call_tool(
            "cap_move_to_joints",
            {
                "arm": "left",
                "target_deg": [0.0] * 7,
                "expected_start_deg": [0.0] * 7,
                "speed_percent": 1,
            },
        )
    )
    content = result[0] if isinstance(result, tuple) else result
    payload = json.loads(next(c.text for c in content if c.type == "text"))
    assert payload["status"] == "measured_endpoint_reached"
    assert payload["observation_available"] is False
    assert payload["replay_motion"] is False
    assert payload["next_required_tool"] == "cap_observe"
    assert len(calls) == 1
    assert events[0]["event"] == "post_motion_observation_failed"
