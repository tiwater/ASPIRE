import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cap_harness.realman.platform import (
    install_recessed_platform,
    is_machine_support_geom,
    nominal_seated_height,
    support_height,
)

mj = pytest.importorskip("mujoco")


def fixture_model():
    config = json.loads(Path("configs/realman/coffee_scene.json").read_text())
    root = ET.Element("mujoco")
    ET.SubElement(root, "option", timestep="0.001", ccd_tolerance="1e-8")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name="coffee_machine")
    ET.SubElement(body, "geom", name="machine_platform", type="box", size=".2 .1575 .0225")
    ET.SubElement(
        body, "geom", name="hotplate", type="cylinder", size=".06 .001", rgba=".1 .1 .1 1"
    )
    install_recessed_platform(asset, body, config["machine"], config["pot"])
    return root, config


def test_recess_is_open_and_ramp_has_correct_height_in_actual_collision_meshes():
    root, cfg = fixture_model()
    m = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    # mj_ray skips invisible geometry: expose collision faces only for this probe.
    m.geom_rgba[m.geom("hotplate").id, 3] = 1
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    machine = cfg["machine"]
    bx, by = machine["pot_bay_x_m"], machine["pot_bay_y_m"]
    # Probe through both cardinal directions and mid-sector rays, not only vertices.
    for angle in [0.0, 0.017, np.pi / 4, 2.1]:
        for r in [0.0, 0.055, 0.06, 0.061, 0.064, 0.067, 0.068, 0.075, 0.077, 0.080]:
            gid = np.array([-1], dtype=np.int32)
            distance = mj.mj_ray(
                m,
                d,
                np.array([bx + r * np.cos(angle), by + r * np.sin(angle), 0.1]),
                np.array([0.0, 0.0, -1.0]),
                np.array([1, 0, 1, 1, 1, 1], dtype=np.uint8),
                1,
                -1,
                gid,
            )
            assert distance >= 0
            assert 0.1 - distance == pytest.approx(support_height(machine, r), abs=3e-5)
    assert 0.1 - mj.mj_ray(
        m,
        d,
        np.array([bx, by, 0.1]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1, 0, 1, 1, 1, 1], dtype=np.uint8),
        1,
        -1,
        np.array([-1], dtype=np.int32),
    ) == pytest.approx(0.043)
    assert nominal_seated_height(machine, cfg["pot"]) == pytest.approx(0.043)
    wider = cfg["pot"] | {"radius_m": 0.080}
    assert nominal_seated_height(machine, wider) == pytest.approx(0.045)
    smaller = cfg["pot"] | {"radius_m": 0.05}
    assert nominal_seated_height(machine, smaller) == pytest.approx(0.043)


def test_invalid_profiles_and_non_support_housing_are_rejected():
    _, cfg = fixture_model()
    for key, value in [
        ("depth_m", -0.002),
        ("ramp_width_m", float("nan")),
        ("segments", 63),
        ("radial_bands", 1),
    ]:
        machine = copy.deepcopy(cfg["machine"])
        machine["hotplate_recess"][key] = value
        with pytest.raises(ValueError):
            support_height(machine, 0)
    assert is_machine_support_geom("machine_platform_ramp_00_001")
    assert not is_machine_support_geom("machine_head")
    assert not is_machine_support_geom("machine_platform_wall")
    legacy = cfg["machine"].copy()
    legacy.pop("hotplate_recess")
    assert nominal_seated_height(legacy, cfg["pot"]) == pytest.approx(0.046)


def test_operator_width_constraint_cannot_regress_to_disk_smaller_than_floor():
    root, cfg = fixture_model()
    cfg["machine"]["hotplate_radius_m"] = 0.060
    with pytest.raises(ValueError, match="must not be smaller"):
        install_recessed_platform(
            root.find("asset"),
            root.find(".//body[@name='coffee_machine']"),
            cfg["machine"],
            cfg["pot"],
        )


def test_nearly_parallel_floor_contacts_do_not_create_false_deep_penetration():
    root, cfg = fixture_model()
    # Also verify reinstalling a generated platform remains a valid operation.
    install_recessed_platform(
        root.find("asset"), root.find(".//body[@name='coffee_machine']"), cfg["machine"], cfg["pot"]
    )
    machine = cfg["machine"]
    pot = ET.SubElement(
        root.find("worldbody"),
        "body",
        name="pot",
        pos=f"{machine['pot_bay_x_m']} {machine['pot_bay_y_m']} .0444999",
    )
    ET.SubElement(pot, "freejoint")
    ET.SubElement(pot, "geom", name="pot_bottom", type="cylinder", size=".068 .0015", mass=".795")
    m = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mj.MjData(m)
    for tilt in [0, 1e-8, 1e-6]:
        d.qpos[3:7] = [np.cos(tilt / 2), np.sin(tilt / 2), 0, 0]
        mj.mj_forward(m, d)
        contacts = [
            c
            for c in d.contact[: d.ncon]
            if {m.geom(c.geom1).name, m.geom(c.geom2).name} == {"hotplate", "pot_bottom"}
        ]
        assert contacts
        for c in contacts:
            assert -c.dist < 1e-5
            assert c.frame[2] * (1 if m.geom(c.geom2).name == "pot_bottom" else -1) > 0.99
