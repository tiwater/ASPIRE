import json
from pathlib import Path

import numpy as np
import pytest

from cap_harness.realman.estimates import estimate_parameters
from cap_harness.realman.payload import (
    AxisymmetricCavity,
    MassProperties,
    body_mass_properties,
    coffee_cavities,
    combine_mass_properties,
    refresh_inertial_constants,
    set_body_properties,
)


def test_cylinder_matches_analytic_volume_center_and_inertia():
    radius, height, density = 0.05, 0.08, 1000
    cavity = AxisymmetricCavity([[0.01, radius], [0.01 + height, radius]], density)
    half = cavity.properties(cavity.capacity_m3 / 2)
    mass = np.pi * radius**2 * (height / 2) * density
    assert half.mass_kg == pytest.approx(mass)
    np.testing.assert_allclose(half.center_m, [0, 0, 0.01 + height / 4])
    np.testing.assert_allclose(
        half.inertia_kg_m2,
        np.diag(
            [
                mass * (3 * radius**2 + (height / 2) ** 2) / 12,
                mass * (3 * radius**2 + (height / 2) ** 2) / 12,
                mass * radius**2 / 2,
            ]
        ),
        atol=1e-12,
    )
    assert cavity.properties(0).mass_kg == 0
    for bad in [-1, float("nan"), cavity.capacity_m3 * 1.01]:
        with pytest.raises(ValueError):
            cavity.properties(bad)


def test_cavities_match_existing_volume_targets_and_parallel_axis_composition():
    cfg = json.loads(Path("configs/realman/coffee_scene.json").read_text())
    cavities = coffee_cavities(cfg)
    estimated = estimate_parameters(cfg)
    for name, key in [("pot", "pot_capacity_ml"), ("cup_0", "cup_capacity_ml")]:
        assert cavities[name].capacity_m3 * 1e6 == pytest.approx(estimated[key])
    left = MassProperties(1, np.array([-1.0, 0, 0]), np.eye(3))
    right = MassProperties(1, np.array([1.0, 0, 0]), np.eye(3))
    merged = combine_mass_properties(left, right)
    np.testing.assert_allclose(merged.center_m, 0)
    np.testing.assert_allclose(merged.inertia_kg_m2, np.diag([2, 4, 4]))


def test_runtime_update_matches_recompiled_inertia_and_preserves_live_state():
    mj = pytest.importorskip("mujoco")
    # Off-center geometry gives a general body while retaining BODYROT shortcuts.
    xml = """<mujoco><worldbody><body name="vessel" pos="0 0 1"><freejoint/>
      <geom name="wall" type="box" size=".03 .04 .05" pos=".01 0 0" mass=".4"/>
      <site name="tip" pos=".02 0 .1"/>
    </body></worldbody></mujoco>"""
    m = mj.MjModel.from_xml_string(xml)
    d = mj.MjData(m)
    d.qpos[:3] = [0.3, -0.2, 1.4]
    d.qvel[:] = np.arange(6) * 0.01
    d.time = 7.25
    mj.mj_forward(m, d)
    poses = (d.geom_xpos.copy(), d.site_xpos.copy())
    q = d.qpos.copy()
    v = d.qvel.copy()
    dry = body_mass_properties(m, "vessel")
    filled = combine_mass_properties(
        dry, MassProperties(1.0, np.array([0, 0, 0.04]), np.diag([0.003, 0.004, 0.002]))
    )
    set_body_properties(m, "vessel", filled)
    refresh_inertial_constants(m, d)
    assert d.time == 7.25
    np.testing.assert_array_equal(d.qpos, q)
    np.testing.assert_array_equal(d.qvel, v)
    np.testing.assert_allclose(d.geom_xpos, poses[0])
    np.testing.assert_allclose(d.site_xpos, poses[1])
    np.testing.assert_allclose(d.xipos[m.body("vessel").id], q[:3] + filled.center_m)
    np.testing.assert_allclose(
        body_mass_properties(m, "vessel").inertia_kg_m2, filled.inertia_kg_m2, atol=1e-12
    )
    # Independent compiler is the reference for mass matrix and gravity terms.
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    inertia = filled.inertia_kg_m2
    ET.SubElement(
        root.find(".//body"),
        "inertial",
        mass=str(filled.mass_kg),
        pos=" ".join(map(str, filled.center_m)),
        fullinertia=" ".join(
            map(
                str,
                [
                    inertia[0, 0],
                    inertia[1, 1],
                    inertia[2, 2],
                    inertia[0, 1],
                    inertia[0, 2],
                    inertia[1, 2],
                ],
            )
        ),
    )
    other = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    od = mj.MjData(other)
    od.qpos[:] = q
    od.qvel[:] = v
    mj.mj_forward(other, od)
    np.testing.assert_allclose(d.qM, od.qM, atol=1e-12)
    np.testing.assert_allclose(d.qfrc_bias, od.qfrc_bias, atol=1e-12)


def test_prescribed_transfer_conserves_mass_and_does_not_claim_real_volume():
    from copy import deepcopy

    mj = pytest.importorskip("mujoco")
    from cap_harness.realman.payload import PrescribedPayloadTransfer

    cfg = json.loads(Path("configs/realman/coffee_scene.json").read_text())
    m = mj.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="pot" pos="0 0 1"><freejoint/>
        <geom type="box" size=".03 .04 .05" pos=".01 0 0" mass=".396"/></body>
      <body name="cup_0" pos=".3 0 1"><freejoint/>
        <geom type="box" size=".02 .02 .04" pos=".001 0 0" mass=".006"/></body>
    </worldbody></mujoco>""")
    d = mj.MjData(m)
    dry = {n: body_mass_properties(m, n) for n in ["pot", "cup_0"]}
    cavities = coffee_cavities(cfg)
    set_body_properties(
        m, "pot", combine_mass_properties(dry["pot"], cavities["pot"].properties(0.001))
    )
    refresh_inertial_constants(m, d)
    transfer = PrescribedPayloadTransfer(m, d, cfg, {n: p.as_dict() for n, p in dry.items()}, 0.001)
    initial_payload = transfer.state()
    total = m.body_mass.sum()
    before = d.qpos.copy()
    for _ in range(80):
        state = transfer.transfer(0.000002, geometry_eligible=True)
    assert state["cup_volume_m3"] == pytest.approx(0.00016)
    assert state["pot_volume_m3"] == pytest.approx(0.00084)
    assert m.body_mass.sum() == pytest.approx(total)
    np.testing.assert_array_equal(d.qpos, before)
    assert not state["actual_volume_observed"] and not state["spill_free_verified"]
    checkpoint = deepcopy(state)
    d.time = 19.5
    d.qvel[:] = np.arange(m.nv) * 0.01
    qvel = d.qvel.copy()
    transfer.restore_payload_state(initial_payload)
    assert transfer.cup_volume_m3 == 0
    restored = transfer.restore_payload_state(checkpoint)
    assert restored["cup_volume_m3"] == checkpoint["cup_volume_m3"]
    assert not restored["actual_volume_observed"] and not restored["spill_free_verified"]
    assert d.time == 19.5
    np.testing.assert_array_equal(d.qpos, before)
    np.testing.assert_array_equal(d.qvel, qvel)
    assert m.body_mass.sum() == pytest.approx(total)
    for change in ["volume", "initial", "mass", "inertia", "claim", "capacity"]:
        bad = deepcopy(checkpoint)
        if change == "volume":
            bad["pot_volume_m3"] += 0.00001
        elif change == "initial":
            bad["initial_volume_m3"] += 0.00001
        elif change == "mass":
            bad["body_properties"]["cup_0"]["mass_kg"] += 0.1
        elif change == "inertia":
            bad["body_properties"]["cup_0"]["inertia_kg_m2"][0][0] = float("nan")
        elif change == "claim":
            bad["spill_free_verified"] = True
        else:
            bad["cup_volume_m3"] = cavities["cup_0"].capacity_m3 * 2
            bad["pot_volume_m3"] = bad["initial_volume_m3"] - bad["cup_volume_m3"]
        bodies = {n: body_mass_properties(m, n).as_dict() for n in ["pot", "cup_0"]}
        with pytest.raises(ValueError):
            transfer.restore_payload_state(bad)
        assert {n: body_mass_properties(m, n).as_dict() for n in bodies} == bodies
        assert transfer.cup_volume_m3 == checkpoint["cup_volume_m3"]
    saved = m.body_mass.copy()
    for amount, eligible in [(0.00005, True), (0.000001, False), (-0.1, True)]:
        with pytest.raises(ValueError):
            transfer.transfer(amount, geometry_eligible=eligible)
        np.testing.assert_array_equal(m.body_mass, saved)


def test_invalid_inertia_update_does_not_partially_change_model():
    mj = pytest.importorskip("mujoco")
    m = mj.MjModel.from_xml_string("""<mujoco><worldbody><body name="vessel">
      <freejoint/><geom type="box" size=".03 .04 .05" pos=".01 0 0" mass=".4"/>
    </body></worldbody></mujoco>""")
    original = body_mass_properties(m, "vessel")
    bad = [
        MassProperties(float("nan"), np.zeros(3), np.eye(3)),
        MassProperties(1, np.zeros(3), np.diag([1.0, 1.0, 3.0])),
        MassProperties(
            1, np.zeros(3), np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ),
    ]
    for properties in bad:
        with pytest.raises(ValueError):
            set_body_properties(m, "vessel", properties)
        assert body_mass_properties(m, "vessel").as_dict() == original.as_dict()


def test_payload_inertia_frame_change_keeps_real_contacts_in_collision_tree():
    mj = pytest.importorskip("mujoco")
    import xml.etree.ElementTree as ET

    xml = """<mujoco><option gravity="0 0 0"/><worldbody>
      <geom name="finger" type="sphere" size=".012" pos=".1 .03 1"/>
      <body name="vessel" pos="0 0 1"><freejoint/>
        <geom name="one" type="box" size=".02 .02 .02" pos=".1 0 0" mass=".2"/>
        <geom name="two" type="box" size=".02 .02 .02" pos="-.08 0 0" mass=".2"/>
      </body></worldbody></mujoco>"""
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    assert data.ncon > 0 and model.body_bvhnum[model.body("vessel").id] > 1
    props = MassProperties(1.2, np.array([0.2, 0.02, 0.03]), np.diag([0.004, 0.005, 0.006]))
    set_body_properties(model, "vessel", props)
    refresh_inertial_constants(model, data)
    tree = ET.fromstring(xml)
    ET.SubElement(
        tree.find(".//body[@name='vessel']"),
        "inertial",
        mass="1.2",
        pos=".2 .02 .03",
        diaginertia=".004 .005 .006",
    )
    other = mj.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    od = mj.MjData(other)
    mj.mj_forward(other, od)
    assert data.ncon == od.ncon > 0
    assert [(c.geom1, c.geom2) for c in data.contact] == [(c.geom1, c.geom2) for c in od.contact]
    np.testing.assert_allclose(data.qacc, od.qacc, rtol=1e-10, atol=1e-9)
    # Repeated updates must not inflate cached bounds; always refit from geometry.
    bounds = model.bvh_aabb.copy()
    for _ in range(20):
        set_body_properties(model, "vessel", props)
    np.testing.assert_allclose(model.bvh_aabb, bounds, atol=1e-12)
