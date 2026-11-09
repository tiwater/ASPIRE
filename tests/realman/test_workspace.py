"""Geometric probes must reject blocked paths and leave the simulation untouched."""

import numpy as np
import pytest
import test_sim as sim_fixtures

from cap_harness.realman.grasp_geometry import grasp_geometry
from cap_harness.realman.joint_planning import plan_joint_path
from cap_harness.realman.workspace import check_joint_segment, pregrasp_targets

built = sim_fixtures.built
adapter = sim_fixtures.adapter


def test_segment_and_planner_keep_state_and_reject_limits(adapter):
    qpos = adapter.data.qpos.copy()
    target = qpos[adapter.qadr["right"]].copy()
    target[-1] = 0.02
    route = plan_joint_path(adapter, "right", target, iterations=1)
    assert route["found"] and not route["physical_motion_allowed"]
    assert check_joint_segment(adapter, {"right": target})["sampled_collision_free"]
    np.testing.assert_array_equal(qpos, adapter.data.qpos)
    assert adapter.data.time == 0
    with pytest.raises(ValueError):
        plan_joint_path(adapter, "right", [99.0] * 7)


def test_target_geometry_is_object_relative_and_has_orientation(adapter):
    targets = pregrasp_targets(adapter)
    for arm in adapter.arms:
        grasp = np.array(targets[arm]["T_world_grasp"])
        np.testing.assert_allclose(grasp[:3, 2], [1, 0, 0])
        assert targets[arm]["oracle_geometry"] and not targets[arm]["grasp_verified"]
    old = np.array(targets["right"]["T_world_grasp"])[:3, 3]
    address = adapter.model.joint("cup_0_free").qposadr[0]
    adapter.data.qpos[address] += 0.01
    adapter.mj.mj_forward(adapter.model, adapter.data)
    new = np.array(pregrasp_targets(adapter)["right"]["T_world_grasp"])[:3, 3]
    np.testing.assert_allclose(new - old, [0.01, 0, 0], atol=1e-12)


def test_right_pregrasp_already_places_camera_above_jaws(adapter):
    targets = pregrasp_targets(adapter)
    target = np.asarray(targets["right"]["T_world_grasp"])
    for arm in adapter.arms:
        initial = np.asarray(grasp_geometry(adapter.model, adapter.data, arm)["T_world_grasp"])
        camera_local = initial[:3, :3].T @ (
            adapter.data.body(arm[0] + "_camera_link").xpos - initial[:3, 3]
        )
        camera_offset = np.asarray(targets[arm]["T_world_grasp"])[:3, :3] @ camera_local
        assert camera_offset[2] > 0.02
        assert targets[arm]["wrist_camera_side"] == "above"
    np.testing.assert_allclose(target[:3, 2], [1, 0, 0])
    np.testing.assert_allclose(target[:3, 1], [0, 0, -1])
    # Left-arm pot approach keeps its original orientation.
    np.testing.assert_allclose(np.asarray(targets["left"]["T_world_grasp"])[:3, 1], [0, 0, 1])


def test_collision_at_start_is_not_ignored(adapter):
    # Move the free pot onto an actual pad to construct a collision negative.
    center = adapter.data.site("right_left_pad_surface").xpos.copy()
    address = adapter.model.joint("pot_free").qposadr[0]
    adapter.data.qpos[address : address + 3] = center - [0, 0, 0.07]
    adapter.mj.mj_forward(adapter.model, adapter.data)
    target = adapter.data.qpos[adapter.qadr["right"]].copy()
    assert not check_joint_segment(adapter, {"right": target})["sampled_collision_free"]
    route = plan_joint_path(adapter, "right", target, iterations=1)
    assert not route["found"] and route["reason"] == "start_or_goal_collision"
