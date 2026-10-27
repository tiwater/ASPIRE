from pathlib import Path

import numpy as np
import pytest

from cap_harness.contracts import ArmCommand, RobotAction, Trajectory
from cap_harness.realman.ik import solve_mount_ik
from cap_harness.realman.model import build_model
from cap_harness.realman.sim import CoffeeSimAdapter

mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    urdf = Path("ref/realman_robot/LSRX_V0.0.13.SLDASM/urdf/LSRX_V0.0.13.SLDASM.urdf")
    if not urdf.exists():
        pytest.skip("requires user-supplied RealBot URDF/meshes")
    out = tmp_path_factory.mktemp("realman_model")
    return build_model(urdf, Path("configs/realman/coffee_scene.json"), out)


@pytest.fixture
def adapter(built):
    a = CoffeeSimAdapter(built["scene"], width=160, height=120)
    yield a
    a.close()


def test_import_preserves_seven_axes_and_gripper_mimics(built, adapter):
    assert built["actuators"] == 16
    assert built["mimic_constraints"] == 10
    assert "l_camera_rojoint" in built["frozen_urdf_joints"]
    state = adapter.get_robot_state()
    assert state.embodiment == "realman"
    assert all(len(q) == 7 for q in state.joint_positions.values())
    assert state.gripper_positions["left"] == pytest.approx(1)
    assert adapter.model.body("pot").mass[0] == pytest.approx(0.396, abs=1e-8)


def test_cup_is_hollow(adapter):
    m, d = adapter.model, adapter.data
    pos = d.body("cup_3").xpos + [0, 0, 0.09]
    gid = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(m, d, pos, np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
    assert 0.08 < distance < 0.095
    assert m.geom(int(gid[0])).name == "cup_3_bottom"


def test_nominal_stack_does_not_fall_off_dispenser(adapter):
    before = {
        n: adapter.data.body(n).xpos.copy() for n in ["pot", "cup_0", "cup_1", "cup_2", "cup_3"]
    }
    adapter._advance(10.0)
    assert not adapter.data.warning.number.any()
    for n, old in before.items():
        assert np.linalg.norm(adapter.data.body(n).xpos - old) < 0.01
    assert adapter.data.body("cup_0").xpos[2] > 0.82


def test_invalid_batch_is_rejected_before_any_control_write(adapter):
    old = adapter.data.ctrl.copy()
    action = RobotAction(
        {
            "left": ArmCommand("joint_position", np.zeros(7), embodiment="realman"),
            "right": ArmCommand("joint_position", np.ones(7) * 99, embodiment="realman"),
        }
    )
    assert not adapter.step(action).ok
    np.testing.assert_array_equal(old, adapter.data.ctrl)


def test_stale_trajectory_does_not_run(adapter):
    state = adapter.get_robot_state()
    t = Trajectory(
        np.zeros((1, 7)),
        0.02,
        state.joint_names["right"],
        "test",
        False,
        state,
        arm="right",
        embodiment="realman",
    )
    adapter.data.qpos[adapter.qadr["left"][0]] += 0.1
    adapter.mj.mj_forward(adapter.model, adapter.data)
    result = adapter.execute_trajectory(t)
    assert not result.ok and result.error.code.value == "stale_plan"
    assert adapter.data.time == 0


def test_ik_is_offline_and_does_not_move_sim_state(adapter):
    old = adapter.data.qpos.copy()
    pose = adapter.get_robot_state().end_effector_poses["right"]
    result = solve_mount_ik(adapter, "right", pose.position, pose.quaternion_wxyz, restarts=1)
    assert result["pose_solved"]
    assert not result["path_checked"]
    assert not result["physical_motion_allowed"]
    np.testing.assert_array_equal(adapter.data.qpos, old)


def test_cap_program_runs_through_public_tools(adapter):
    from cap_harness.api import CapApi
    from cap_harness.run import run_program_with_api

    result = run_program_with_api(
        api=CapApi(adapter, default_camera="front"),
        program_path=Path("examples/realman/observe_and_move_sim.py"),
    )
    assert result.ok
    assert result.result["motion_ok"]
    assert result.result["joints_per_arm"] == 7
    assert any(call.tool == "move_to_joints" for call in result.calls)
    assert adapter.get_robot_state().joint_positions["right"][6] == pytest.approx(0.03, abs=0.01)


def test_frozen_body_pose_changes_fk_without_adding_actuators(tmp_path, built):
    import json

    config = json.loads(Path("configs/realman/coffee_scene.json").read_text())
    config["robot"]["frozen_joints_rad"] = {"xt_joint_1": -0.2}
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps(config))
    result = build_model(
        Path("ref/realman_robot/LSRX_V0.0.13.SLDASM/urdf/LSRX_V0.0.13.SLDASM.urdf"),
        scene,
        tmp_path / "model",
    )
    a = CoffeeSimAdapter(result["scene"])
    baseline = CoffeeSimAdapter(built["scene"])
    try:
        assert result["actuators"] == 16
        assert result["frozen_urdf_positions_rad"]["xt_joint_1"] == -0.2
        assert (
            np.linalg.norm(a.data.body("xb_link").xpos - baseline.data.body("xb_link").xpos) > 0.1
        )
    finally:
        a.close()
        baseline.close()
