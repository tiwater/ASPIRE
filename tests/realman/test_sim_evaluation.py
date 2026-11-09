"""Counterfactual fixture edits test the grader, never count as policy rollouts."""

import json
from pathlib import Path

import test_sim as sim_fixtures

from cap_harness.realman.sim_evaluation import CoffeeGeometryEvaluator

built = sim_fixtures.built
adapter = sim_fixtures.adapter


def config():
    return json.loads(Path("configs/realman/coffee_scene.json").read_text())


def evidence(adapter):
    return {k: v.value for k, v in adapter.task_evidence["evidence"].items()}


def test_initial_fixture_is_not_a_grasp_or_coffee_success(adapter):
    adapter.enable_task_evaluation(config())
    adapter._advance(0.8)
    e = evidence(adapter)
    # Initially only cup_0 bears on the rack; the others load through cup_0.
    # This is not yet independent retention after extracting that cup.
    assert not e["remaining_cups_retained"]
    assert not e["cup_in_right_gripper"] and not e["pot_handle_in_left_gripper"]
    assert not e["single_cup_separated"] and not e["cup_supported_by_table"]
    assert e["fill_target_reached"] is None and e["no_spill"] is None
    assert adapter.task_evidence["coffee_task_success"] is None


def test_table_support_requires_force_and_a_temporal_window(adapter):
    cfg = config()
    adr = adapter.model.joint("cup_0_free").qposadr[0]
    # Place a fixture over free tabletop, then let gravity establish support.
    adapter.data.qpos[adr : adr + 3] = [0.70, -0.40, 0.757]
    adapter.mj.mj_forward(adapter.model, adapter.data)
    adapter.enable_task_evaluation(cfg)
    assert not evidence(adapter)["cup_supported_by_table"]
    adapter._advance(0.8)
    assert evidence(adapter)["cup_supported_by_table"]
    assert evidence(adapter)["right_gripper_released"]
    # Same upright XY in mid-air is not a supported placement.
    adapter.data.qpos[adr + 2] += 0.10
    adapter.mj.mj_forward(adapter.model, adapter.data)
    result = adapter.task_evaluator.update()
    assert not result["evidence"]["cup_supported_by_table"].value
    assert not result["evidence"]["right_gripper_released"].value


def test_multi_cup_removal_and_long_observation_gap_fail(adapter):
    adapter.enable_task_evaluation(config())
    adapter._advance(0.8)
    for name in ["cup_0", "cup_1"]:
        adr = adapter.model.joint(name + "_free").qposadr[0]
        adapter.data.qpos[adr : adr + 3] += [-0.2, -0.2, 0.1]
    adapter.mj.mj_forward(adapter.model, adapter.data)
    result = adapter.task_evaluator.update()
    assert not result["evidence"]["remaining_cups_retained"].value
    assert not result["evidence"]["single_cup_separated"].value
    # A long unobserved interval cannot turn one sample into stable support.
    ev = CoffeeGeometryEvaluator(adapter, config())
    ev.update()
    adapter.data.time += 1
    assert not ev.update()["evidence"]["pot_stable_after_release"].value


def test_alignment_uses_raised_lid_but_still_rejects_lateral_offset(adapter):
    adapter.enable_task_evaluation(config())
    adapter._advance(0.8)
    assert evidence(adapter)["lid_outlet_aligned"]
    adr = adapter.model.joint("pot_free").qposadr[0]
    adapter.data.qpos[adr] += 0.008
    adapter.mj.mj_forward(adapter.model, adapter.data)
    assert not adapter.task_evaluator.update()["evidence"]["lid_outlet_aligned"].value


def test_support_grace_cannot_acquire_missing_support_or_hide_multi_cup_removal(adapter):
    cfg = config()
    cfg["simulation_diagnostics"] = {
        "rack_support": {"maximum_gap_s": 0.04, "maximum_excursion_m": 0.0005}
    }
    adapter.enable_task_evaluation(cfg)
    adapter._advance(0.8)
    assert not evidence(adapter)["remaining_cups_retained"]
    ev = adapter.task_evaluator
    # Counterfactual prior establishment must not override new geometric failure.
    ev.retention_established = True
    for name in ev.cups[1:]:
        ev.support_continuity.update(
            name,
            float(adapter.data.time),
            adapter.data.body(name).xpos,
            anchored=True,
            geometry_valid=True,
            established=True,
        )
    for name in ["cup_0", "cup_1"]:
        adr = adapter.model.joint(name + "_free").qposadr[0]
        adapter.data.qpos[adr : adr + 3] += [-0.2, -0.2, 0.1]
    adapter.mj.mj_forward(adapter.model, adapter.data)
    result = ev.update()
    assert not result["evidence"]["remaining_cups_retained"].value
    assert not result["evidence"]["single_cup_separated"].value
    assert not result["rack_support_diagnostics"]["cups"]["cup_1"]["geometry_valid"]
