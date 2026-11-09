"""Independent pickup audit rejects unrelated state changes and false delivery."""
import copy
import importlib
from pathlib import Path

import pytest

from test_service_collection import fixture
from cap_harness.realman.service_collection import StaffCupCollection


@pytest.fixture
def auditor(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools/realman"))
    return importlib.import_module("audit_cup_delivery")


def transition():
    a, config, payload, _ = fixture()
    config["table"]["center_xy_m"][0] = 1.09  # fixed delivery x=.65
    c = StaffCupCollection(a, config, payload, simulation_only=True)
    before = copy.deepcopy(a.data)
    bm = dict(time_s=before.time, staff_collection=c.state(), runtime_payload=payload.state(),
              runtime_grip={"kp": 36}, runtime_options={"timestep": .001}, grasp_context={"epoch": 1})
    c.collect("cup_0", staff_confirmed=True)
    am = {**copy.deepcopy(bm), "staff_collection": c.state()}
    return a.model, before, a.data, bm, am, config


def test_actual_physical_pickup_transition_passes(auditor):
    assert auditor.inspect_transition(*transition())["passed"]


@pytest.mark.parametrize("change,failed", [
    ("other_position", "other_objects_and_robot_unchanged"),
    ("other_velocity", "other_objects_and_robot_unchanged"),
    ("solver_jump", "solver_cache_unchanged_within_roundoff"),
    ("payload", "payload_and_grasp_preserved"),
    ("grasp", "payload_and_grasp_preserved"),
    ("time", "same_clock"),
    ("pose", "declared_after_pose"),
    ("slot", "fixed_delivery_point"),
])
def test_mutated_checkpoint_cannot_pass_despite_success_receipt(auditor, change, failed):
    m, before, after, bm, am, config = transition()
    if change == "other_position": after.qpos[7] += .001
    elif change == "other_velocity": after.qvel[6] += .001
    elif change == "solver_jump": after.qacc_warmstart[6] += .01
    elif change == "payload": am["runtime_payload"]["vessel_volumes_m3"]["pot"] += .0001
    elif change == "grasp": am["grasp_context"]["epoch"] += 1
    elif change == "time": after.time += .02
    elif change == "pose": after.qpos[0] += .01
    elif change == "slot": config["table"]["center_xy_m"][0] += .1
    report = auditor.inspect_transition(m, before, after, bm, am, config)
    assert not report["passed"] and not report["checks"][failed]


def test_duplicate_external_event_is_rejected(auditor):
    m, before, after, bm, am, config = transition()
    bm["staff_collection"] = copy.deepcopy(am["staff_collection"])
    am["staff_collection"]["events"].append(copy.deepcopy(am["staff_collection"]["events"][0]))
    report = auditor.inspect_transition(m, before, after, bm, am, config)
    assert not report["passed"] and not report["checks"]["explicit_staff_event"]


def test_only_operator_changed_final_relation_is_replaced(monkeypatch, tmp_path, auditor):
    import json
    module = importlib.import_module("audit_coffee_delivery_session")
    predicates = {k: True for k in [
        "pot_on_hotplate", "pot_stable_after_release", "lid_outlet_aligned", "pot_upright",
        "left_gripper_released", "remaining_cups_retained", "previous_served_cups_preserved",
        "both_grippers_empty", "served_cups_collected",
    ]}
    params = dict(delivery_mode="staff_collect_fixed_slot", staff_pickup_assumed=True,
                  external_actor_object_pose_writes=True, staff_collection={"simulation_only": True})
    (tmp_path / "result.json").write_text(json.dumps(dict(predicates=predicates, parameters=params)))
    monkeypatch.setattr(module, "audit_session", lambda *a, **k: {
        "checks": {"final_object_relations": False, "guard_trace": False}, "passed": False})
    monkeypatch.setattr(module, "audit_delivery", lambda *a, **k: {"passed": True})
    report = module.audit(tmp_path)
    assert report["checks"]["final_object_relations"]
    assert not report["checks"]["guard_trace"] and not report["passed"]
