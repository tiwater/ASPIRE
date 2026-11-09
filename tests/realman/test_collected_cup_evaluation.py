"""Counterfactual grader tests: collection must not mask actual object drops."""
import json
from pathlib import Path

import pytest
import test_sim as sim_fixtures

from cap_harness.realman.service_evaluation import ServiceGeometryEvaluator

built = sim_fixtures.built
adapter = sim_fixtures.adapter


@pytest.mark.parametrize("case,expected", [
    ("collected", []),
    ("uncollected", ["cup_0"]),
    ("unserved", ["cup_0"]),
    ("returned_to_workspace", ["cup_0"]),
    ("pot_dropped", ["pot"]),
    ("other_cup_dropped", ["cup_1"]),
    ("previous_served_dropped", ["cup_1"]),
])
def test_drop_detection_respects_only_explicit_completed_collection(adapter, case, expected):
    cfg = json.loads(Path("configs/realman/coffee_scene.json").read_text())
    ev = ServiceGeometryEvaluator(adapter, cfg)
    d, m, mj = adapter.data, adapter.model, adapter.mj
    ev.served["cup_0"] = d.body("cup_0").xpos.copy()
    ev.collected_cups["cup_0"] = [-1.5, -1.5, .0001]
    def place(name, pos):
        adr = m.joint(name + "_free").qposadr[0]
        d.qpos[adr:adr+3] = pos
    place("cup_0", [-1.5, -1.5, .0001])
    if case == "uncollected":
        ev.collected_cups.clear()
    elif case == "unserved":
        ev.served.clear()
    elif case == "returned_to_workspace":
        place("cup_0", [*cfg["table"]["center_xy_m"], .01])
    elif case == "pot_dropped":
        place("pot", [-1.5, -1.8, .01])
    elif case in {"other_cup_dropped", "previous_served_dropped"}:
        if case == "previous_served_dropped":
            ev.served["cup_1"] = d.body("cup_1").xpos.copy()
            ev.cups.remove("cup_1")
        place("cup_1", [-1.5, -1.8, .01])
    mj.mj_forward(m, d)
    result = ev.update()
    assert result["object_height_diagnostics"]["below_table_objects"] == expected
    assert ("object_below_table" in result["failures"]) == bool(expected)
    if case == "returned_to_workspace":
        assert "previous_served_cup_relation_lost" in result["failures"]
