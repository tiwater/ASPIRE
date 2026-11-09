import json
from pathlib import Path

import numpy as np
import pytest

from cap_harness.realman.payload import (
    body_mass_properties,
    coffee_cavities,
    combine_mass_properties,
    refresh_inertial_constants,
    set_body_properties,
)
from cap_harness.realman.service_payload import ServicePayload


def make_payload():
    mj = pytest.importorskip("mujoco")
    cfg = json.loads(Path("configs/realman/coffee_scene.json").read_text())
    cfg["dispenser"]["stack_count"] = 3
    cups = "".join(
        f'<body name="cup_{i}" pos="{0.3 + i * 0.2} 0 1"><freejoint/>'
        '<geom type="box" size=".02 .02 .04" pos=".001 0 0" mass=".006"/></body>'
        for i in range(3)
    )
    m = mj.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="pot" pos="0 0 1">'
        '<freejoint/><geom type="box" size=".03 .04 .05" pos=".01 0 0" mass=".396"/>'
        "</body>" + cups + "</worldbody></mujoco>"
    )
    d = mj.MjData(m)
    dry = {n: body_mass_properties(m, n) for n in ["pot", "cup_0"]}
    cfg["payload"] = {"pot_initial_ml": 400, "dry": {n: p.as_dict() for n, p in dry.items()}}
    set_body_properties(
        m,
        "pot",
        combine_mass_properties(dry["pot"], coffee_cavities(cfg)["pot"].properties(0.0004)),
    )
    refresh_inertial_constants(m, d)
    payload = ServicePayload(m, d, cfg)
    return m, d, payload


def test_distinct_cup_loads_survive_reselection_and_refill_without_pose_reset():
    m, d, payload = make_payload()
    d.time = 17.0
    d.qvel[:] = np.arange(m.nv) * 0.001
    q, v, total = d.qpos.copy(), d.qvel.copy(), float(m.body_mass.sum())
    payload.transfer(0.00016, geometry_eligible=True)
    first_inertia = body_mass_properties(m, "cup_0").inertia_kg_m2.copy()
    payload.select_cup("cup_1")
    payload.transfer(0.00016, geometry_eligible=True)
    assert payload.pot_volume_m3 == pytest.approx(0.00008)
    assert m.body("cup_0").mass[0] == pytest.approx(0.166)
    assert m.body("cup_1").mass[0] == pytest.approx(0.166)
    assert float(m.body_mass.sum()) == pytest.approx(total)
    np.testing.assert_array_equal(body_mass_properties(m, "cup_0").inertia_kg_m2, first_inertia)
    with pytest.raises(ValueError):
        payload.select_cup("cup_0")
    with pytest.raises(ValueError):
        payload.refill(0.00032, staff_confirmed=True)
    payload.refill(0.00032, staff_confirmed=True, pot_supported_released=True)
    payload.select_cup("cup_2")
    payload.transfer(0.00016, geometry_eligible=True)
    state = payload.state()
    assert state["pot_volume_m3"] == pytest.approx(0.00024)
    assert sum(state["vessel_volumes_m3"].values()) == pytest.approx(0.00072)
    assert float(m.body_mass.sum()) == pytest.approx(total + 0.32)
    assert not state["actual_volume_observed"]
    np.testing.assert_array_equal(d.qpos, q)
    np.testing.assert_array_equal(d.qvel, v)
    assert d.time == 17.0
    before = m.body_mass.copy()
    with pytest.raises(ValueError):
        payload.transfer(0.0001, geometry_eligible=True)
    np.testing.assert_array_equal(m.body_mass, before)


def test_restore_multicup_refill_checkpoint_preserves_integration_state():
    source_m, source_d, source = make_payload()
    source.transfer(0.00016, geometry_eligible=True)
    source.select_cup("cup_1")
    source.transfer(0.00016, geometry_eligible=True)
    source_d.time = 12
    source.refill(0.00032, staff_confirmed=True, pot_supported_released=True)
    expected = source.state()
    m, d, payload = make_payload()
    d.time = 31
    d.qpos[0] += 0.04
    d.qvel[:] = np.arange(m.nv) * 0.001
    q, v = d.qpos.copy(), d.qvel.copy()
    restored = payload.restore_payload_state(expected)
    assert restored["vessel_volumes_m3"] == expected["vessel_volumes_m3"]
    assert restored["external_events"] == expected["external_events"]
    np.testing.assert_allclose(m.body_mass, source_m.body_mass, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(d.qpos, q)
    np.testing.assert_array_equal(d.qvel, v)
    assert d.time == 31
    expected["external_events"][0]["added_m3"] = 0
    assert payload.events[0]["added_m3"] == 0.00032


@pytest.mark.parametrize("corruption", ["inertia", "ledger", "conservation", "identity"])
def test_corrupt_checkpoint_rejected_without_partial_load_changes(corruption):
    _, _, source = make_payload()
    source.transfer(0.00016, geometry_eligible=True)
    source.refill(0.0001, staff_confirmed=True, pot_supported_released=True)
    state = source.state()
    if corruption == "inertia":
        state["body_properties"]["cup_2"]["mass_kg"] += 0.001
    elif corruption == "ledger":
        state["external_events"][0]["added_m3"] += 0.0001
    elif corruption == "conservation":
        state["vessel_volumes_m3"]["cup_0"] += 0.0001
    else:
        state["active_cup"] = "cup_9"
    m, d, payload = make_payload()
    mass, inertia, q, v = m.body_mass.copy(), m.body_inertia.copy(), d.qpos.copy(), d.qvel.copy()
    old = payload.state()
    with pytest.raises(ValueError):
        payload.restore_payload_state(state)
    assert payload.state() == old
    np.testing.assert_array_equal(m.body_mass, mass)
    np.testing.assert_array_equal(m.body_inertia, inertia)
    np.testing.assert_array_equal(d.qpos, q)
    np.testing.assert_array_equal(d.qvel, v)
