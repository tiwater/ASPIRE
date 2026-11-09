import copy
from types import SimpleNamespace

import numpy as np
import pytest

from cap_harness.realman.service_collection import StaffCupCollection, fixed_delivery_slot


def fixture():
    mj = pytest.importorskip("mujoco")
    model = mj.MjModel.from_xml_string("""
    <mujoco><option timestep=".001"/><worldbody>
      <geom name="floor" type="plane" size="3 3 .1"/>
      <geom name="tabletop" type="box" pos="1 0 .725" size=".5 .5 .025"/>
      <body name="cup_0" pos=".65 -.09 .75"><freejoint/>
        <geom name="cup_0_shell" type="cylinder" pos="0 0 .04" size=".025 .04" mass=".166"/>
      </body>
      <body name="cup_1" pos="1.1 .2 .75"><freejoint/>
        <geom name="cup_1_shell" type="cylinder" pos="0 0 .04" size=".025 .04" mass=".006"/>
      </body>
      <body name="rtool_base_link" pos="3 0 1"><freejoint/>
        <geom name="rtool_base_link_collision_0" type="box" size=".02 .02 .02" mass="1"/>
      </body>
    </worldbody></mujoco>""")
    data = mj.MjData(model)
    for _ in range(200):
        mj.mj_step(model, data)
    mj.mj_forward(model, data)
    config = {
        "table": {"center_xy_m": [1, 0], "size_m": [1, 1, .05]},
        "cup": {"bottom_radius_m": .025, "rim_radius_m": .035},
        "payload": {"cup_capacity_ml": 200},
        "dispenser": {"stack_count": 4},
    }
    volumes = {"pot": .00024, "cup_0": .00016, "cup_1": 0}
    payload = SimpleNamespace(state=lambda: {"vessel_volumes_m3": volumes.copy()})
    evaluator = SimpleNamespace(served={"cup_0": data.body("cup_0").xpos.copy()}, collected_cups={})
    adapter = SimpleNamespace(model=model, data=data, mj=mj, task_evaluator=evaluator)
    return adapter, config, payload, volumes


def test_pickup_moves_only_completed_cup_and_preserves_loads_and_clock():
    a, config, payload, _ = fixture()
    collector = StaffCupCollection(a, config, payload, simulation_only=True)
    m, d = a.model, a.data
    q, v, mass, ledger, time = d.qpos.copy(), d.qvel.copy(), m.body_mass.copy(), payload.state(), d.time
    event = collector.collect("cup_0", staff_confirmed=True)
    # cup_0 is the first free joint in this small physical fixture.
    np.testing.assert_array_equal(d.qpos[7:], q[7:])
    np.testing.assert_array_equal(d.qvel[6:], v[6:])
    np.testing.assert_array_equal(m.body_mass, mass)
    assert payload.state() == ledger and d.time == time
    assert event["observed_before"]["table_support_force_n"] > .02
    assert event["volume_ml"] == pytest.approx(160)
    assert d.body("cup_0").xpos[0] < -1
    assert set(a.task_evaluator.served) == {"cup_0"}
    assert set(a.task_evaluator.collected_cups) == {"cup_0"}
    recorded = collector.state()
    recorded["events"].clear()
    assert len(collector.events) == 1
    after = d.qpos.copy()
    with pytest.raises(ValueError, match="not previously collected"):
        collector.collect("cup_0", staff_confirmed=True)
    np.testing.assert_array_equal(d.qpos, after)


@pytest.mark.parametrize("failure", ["unconfirmed", "unserved", "unfilled", "unsupported", "moving"])
def test_invalid_pickup_is_rejected_before_any_pose_or_ledger_change(failure):
    a, config, payload, volumes = fixture()
    collector = StaffCupCollection(a, config, payload, simulation_only=True)
    cup = "cup_0"
    if failure == "unserved":
        cup = "cup_1"
    elif failure == "unfilled":
        volumes["cup_0"] = .0001
    elif failure == "unsupported":
        a.data.qpos[2] += .1
    elif failure == "moving":
        a.data.qvel[0] = .1
    a.mj.mj_forward(a.model, a.data)
    q, v, state = a.data.qpos.copy(), a.data.qvel.copy(), copy.deepcopy(payload.state())
    with pytest.raises(ValueError):
        collector.collect(cup, staff_confirmed=failure != "unconfirmed")
    np.testing.assert_array_equal(a.data.qpos, q)
    np.testing.assert_array_equal(a.data.qvel, v)
    assert payload.state() == state and not collector.events and not a.task_evaluator.collected_cups


def test_fixed_delivery_slot_reuses_first_slot_and_requires_simulation_mode():
    a, config, payload, _ = fixture()
    np.testing.assert_allclose(fixed_delivery_slot(config), [.56, -.09])
    config["table"]["center_xy_m"][1] = .2
    np.testing.assert_allclose(fixed_delivery_slot(config), [.56, .11])
    with pytest.raises(ValueError, match="simulation assumption"):
        StaffCupCollection(a, config, payload, simulation_only=False)
