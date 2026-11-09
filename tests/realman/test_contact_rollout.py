"""Force-aware collision supervision and numerical thin-wall contact settings."""

import numpy as np
import pytest

from cap_harness.realman.workspace import robot_contacts

mj = pytest.importorskip("mujoco")


def test_loaded_shallow_contact_is_not_missed_by_penetration_threshold():
    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom name="ground" type="plane" size="1 1 .1"/>
    <body name="base_link" pos="0 0 .01995"><freejoint/>
    <geom name="robot_ball" type="sphere" size=".02" mass="1"/>
    </body></worldbody></mujoco>""")
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    assert not robot_contacts(model, data)
    contacts = robot_contacts(model, data, normal_force_threshold_n=0.02)
    assert contacts and contacts[0]["normal_force_n"] > 0.02
    assert contacts[0]["penetration_m"] < 0.0001


def test_solver_mix_controls_contact_without_changing_friction():
    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom name="cup_wall" type="plane" size="1 1 .1" friction=".5 .005 .0001"/>
    <body name="base_link" pos="0 0 .01995"><freejoint/>
    <geom name="pad" type="sphere" size=".02" mass="1"
      solref=".005 1" solimp=".999 .999 .001" solmix="1000" friction=".8 .005 .0001"/>
    </body></worldbody></mujoco>""")
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    contact = data.contact[0]
    assert contact.solref[0] == pytest.approx((1000 * 0.005 + 0.02) / 1001)
    assert contact.friction[0] == pytest.approx(0.8)
    assert np.isfinite(data.qacc).all()


def test_contact_allowance_never_covers_another_cup_or_entire_support(monkeypatch):
    from types import SimpleNamespace

    import cap_harness.realman.contact_rollout as module
    from cap_harness.realman.contact_rollout import ContactRollout

    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="base_link">
      <geom name="right_left_pad_contact" type="sphere" size=".001"/>
      <geom name="rtool_Left_Support_Link_collision_0" type="sphere" size=".001"/>
      <site name="right_left_pad_surface"/>
    </body>
    <body name="cup_0"><geom name="cup_0_wall" type="sphere" size=".001"/></body>
    <body name="cup_1"><geom name="cup_1_wall" type="sphere" size=".001"/></body>
    </worldbody></mujoco>""")
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    trial = ContactRollout.__new__(ContactRollout)
    trial.adapter = SimpleNamespace(
        model=model,
        data=data,
        arms=["right"],
        _advance=lambda dt: None,
        task_evidence={"failures": []},
    )
    trial.record = lambda: None
    trial.allowed = {("right_left_pad_contact", "cup_0")}
    trial.required_relations = set()
    trial.require_pot_pinch = False
    trial.max_linkage_error_rad = {"right": 0.0}
    trial._stage = "test"
    contact = {
        "geom1": "right_left_pad_contact",
        "geom2": "cup_0_wall",
        "penetration_m": 0.00001,
        "position_world_m": [0, 0, 0],
        "normal_force_n": 0.1,
    }
    monkeypatch.setattr(module, "robot_contacts", lambda *a, **kw: [contact])
    trial.tick()
    contact["geom2"] = "cup_1_wall"
    with pytest.raises(RuntimeError, match="unexpected robot contact"):
        trial.tick()
    contact.update(geom1="rtool_Left_Support_Link_collision_0", geom2="cup_0_wall")
    trial.tick()
    contact["position_world_m"] = [0.03, 0, 0]
    with pytest.raises(RuntimeError, match="unexpected robot contact"):
        trial.tick()


def test_cli_reports_failed_rollout_with_nonzero_exit(monkeypatch, capsys):
    import sys

    from cap_harness.realman import __main__, contact_rollout

    monkeypatch.setattr(
        sys, "argv", ["realman", "contact-rollout", "--review", "review", "--output", "run"]
    )
    monkeypatch.setattr(
        contact_rollout,
        "run_contact_experiment",
        lambda *args, **kwargs: {"experiment_success": False, "coffee_task_success": None},
    )
    with pytest.raises(SystemExit) as caught:
        __main__.main()
    assert caught.value.code == 2
    assert '"experiment_success": false' in capsys.readouterr().out


def test_old_camera_down_review_is_rejected_before_any_path_execution(monkeypatch):
    from types import SimpleNamespace

    from cap_harness.realman import contact_rollout

    def must_not_move():
        pytest.fail("stale review must be rejected before prepare")

    trial = SimpleNamespace(
        plan={"targets": {"right": {}}},
        prepare=must_not_move,
        finish=lambda error: {"error": error},
    )
    monkeypatch.setattr(contact_rollout, "ContactRollout", lambda *a, **kw: trial)
    result = contact_rollout.run_contact_experiment("old_review", "unused")
    assert "rebuild workspace review" in result["error"]


def test_pot_pinch_requires_both_pads_on_handle_not_vessel():
    from cap_harness.realman.contact_rollout import ContactRollout

    trial = ContactRollout.__new__(ContactRollout)
    contacts = [
        {"geom1": "left_left_pad_contact", "geom2": "handle_collision_0", "normal_force_n": 1.0},
        {"geom1": "handle_collision_1", "geom2": "left_right_pad_contact", "normal_force_n": 2.0},
    ]
    trial.contacts = lambda: contacts
    trial.check_pot_pinch()
    contacts[1]["geom1"] = "pot_glass_collision_0"
    with pytest.raises(RuntimeError, match="lost bilateral pot-handle pinch"):
        trial.check_pot_pinch()
    contacts[1]["geom1"] = "handle_collision_1"
    contacts[1]["normal_force_n"] = 0.001
    with pytest.raises(RuntimeError, match="lost bilateral pot-handle pinch"):
        trial.check_pot_pinch()


def test_dispenser_cli_preserves_failed_mechanism_result(monkeypatch, capsys):
    import sys

    from cap_harness.realman import __main__, dispenser_bench

    monkeypatch.setattr(
        sys, "argv", ["realman", "dispenser-bench", "--model", "scene.xml", "--output", "run"]
    )
    monkeypatch.setattr(
        dispenser_bench,
        "run_dispenser_bench",
        lambda *args, **kwargs: {"bench_success": False, "coffee_task_success": None},
    )
    with pytest.raises(SystemExit) as caught:
        __main__.main()
    assert caught.value.code == 2
    assert '"bench_success": false' in capsys.readouterr().out


def test_horizontal_withdrawal_requires_object_clearance_despite_wrist_travel():
    from types import SimpleNamespace

    from cap_harness.realman.contact_rollout import cup_withdraw_clearance

    cups = {
        "cup_0": SimpleNamespace(xpos=np.array([0.0, 0.0, 0.78]), xmat=np.eye(3).ravel()),
        "cup_1": SimpleNamespace(xpos=np.array([0.0, 0.0, 0.85]), xmat=np.eye(3).ravel()),
    }
    data = SimpleNamespace(body=cups.__getitem__)
    config = {
        "cup": {"height_m": 0.08, "rim_radius_m": 0.035, "bottom_radius_m": 0.025},
        "dispenser": {"stack_count": 2},
    }
    assert cup_withdraw_clearance(data, config) == pytest.approx(-0.01)
    cups["cup_0"].xpos[2] = 0.765
    assert cup_withdraw_clearance(data, config) == pytest.approx(0.005)
    # A tilted rim can still overlap, even if the cup origin passed its target.
    cups["cup_0"].xmat[8] = np.cos(np.deg2rad(20))
    assert cup_withdraw_clearance(data, config) < 0


@pytest.mark.parametrize(
    "direction", [[-1, 0, 0], [0, 1, 0], [-0.7, 0.7, 0.1], [-1, 1, 0], [float("nan"), 1, 0]]
)
def test_pot_path_rejects_axial_or_nonhorizontal_withdrawal(direction):
    from cap_harness.realman.pot_bench import diagonal_withdraw_delta

    with pytest.raises(ValueError, match="left-back diagonal"):
        diagonal_withdraw_delta(
            {"task": {"withdraw_table_direction": direction, "withdraw_distance_m": 0.18}}
        )


def test_pot_path_is_robot_left_back_and_reverse_returns_to_seat():
    from cap_harness.realman.pot_bench import diagonal_withdraw_delta

    delta = diagonal_withdraw_delta(
        {
            "task": {
                "withdraw_table_direction": [-(2**-0.5), 2**-0.5, 0],
                "withdraw_distance_m": 0.18,
            }
        }
    )
    assert delta[0] < 0 < delta[1] and delta[2] == 0
    assert np.linalg.norm(delta) == pytest.approx(0.18)


@pytest.mark.parametrize(
    "metadata, expected_error",
    [
        ({"runtime_payload": {"cup_volume_m3": 0.00016}}, "explicit inertia restoration"),
        ({"runtime_options": {"timestep_s": 0.00025}}, "explicit option restoration"),
    ],
)
def test_checkpoint_cannot_silently_restore_different_xml_defaults(
    tmp_path, monkeypatch, metadata, expected_error
):
    import hashlib
    import json
    from types import SimpleNamespace

    import cap_harness.realman.contact_rollout as module

    review = tmp_path / "review"
    (review / "model").mkdir(parents=True)
    model_bytes = b"placeholder; adapter is replaced in this checkpoint contract test"
    (review / "model/scene.xml").write_bytes(model_bytes)
    (review / "result.json").write_text(
        json.dumps({"model_sha256": hashlib.sha256(model_bytes).hexdigest()})
    )
    (review / "scene_config.json").write_text("{}")
    state = tmp_path / "loaded.npz"
    state.with_suffix(".json").write_text(json.dumps(metadata))
    adapter = SimpleNamespace(data=SimpleNamespace(time=0), enable_task_evaluation=lambda cfg: None)
    monkeypatch.setattr(module, "CoffeeSimAdapter", lambda *a, **kw: adapter)
    with pytest.raises(ValueError, match=expected_error):
        module.ContactRollout(review, tmp_path / "output", checkpoint=state)


def test_checkpoint_records_changed_integration_timestep(tmp_path):
    import json
    from types import SimpleNamespace

    from cap_harness.realman.contact_rollout import ContactRollout

    model = mj.MjModel.from_xml_string('<mujoco><option timestep=".001"/></mujoco>')
    data = mj.MjData(model)
    trial = ContactRollout.__new__(ContactRollout)
    trial.adapter = SimpleNamespace(model=model, data=data, mj=mj)
    trial.initial_timestep_s = 0.001
    trial.initial_enable_flags = 0
    trial.initial_disable_flags = 0
    trial.model_hash = "test"
    trial.allowed = set()
    trial._stage = "diagnostic"
    model.opt.timestep = 0.00025
    model.opt.enableflags = int(mj.mjtEnableBit.mjENBL_MULTICCD)
    target = tmp_path / "changed.npz"
    trial.save_checkpoint(target)
    assert json.loads(target.with_suffix(".json").read_text())["runtime_options"] == {
        "timestep_s": 0.00025,
        "enable_flags": int(mj.mjtEnableBit.mjENBL_MULTICCD),
    }
