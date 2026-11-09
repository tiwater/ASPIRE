import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cap_harness.realman.platform import (
    install_convex_pot_floor,
    install_recessed_platform,
    support_height,
)

mj = pytest.importorskip("mujoco")


def fixture(offset=0.0, tilt=0.0):
    cfg = json.loads(Path("configs/realman/coffee_scene.json").read_text())
    machine = cfg["machine"]
    root = ET.Element("mujoco")
    ET.SubElement(root, "option", ccd_tolerance="1e-8")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name="coffee_machine")
    ET.SubElement(body, "geom", name="machine_platform", type="box", size=".2 .1575 .0225")
    ET.SubElement(body, "geom", name="hotplate", type="cylinder", size=".069 .001")
    install_recessed_platform(asset, body, machine, cfg["pot"])
    z = support_height(machine, offset + 0.068) - 1e-7
    pot = ET.SubElement(
        world,
        "body",
        name="pot",
        pos=f"{machine['pot_bay_x_m'] + offset} {machine['pot_bay_y_m']} {z}",
        quat=f"{np.cos(tilt / 2)} {np.sin(tilt / 2)} 0 0",
    )
    ET.SubElement(pot, "freejoint")
    ET.SubElement(pot, "inertial", mass=".795", pos=".009 0 .044", diaginertia=".001 .002 .0025")
    ET.SubElement(
        pot,
        "geom",
        name="pot_bottom",
        type="cylinder",
        size=".068 .0015",
        pos="0 0 .0015",
        friction=".8 .005 .0001",
        solref=".005 1",
        solimp=".999 .999 .001",
    )
    return root, asset, pot


@pytest.mark.parametrize("offset,tilt", [(0.0, 0.0), (0.068, 1e-7), (0.13, 1e-6)])
def test_floor_prism_retains_payload_and_has_consistent_upward_contacts(offset, tilt):
    root, asset, pot = fixture(offset, tilt)
    before = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    info = install_convex_pot_floor(asset, pot)
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for name in ["body_mass", "body_inertia", "body_ipos", "body_iquat"]:
        np.testing.assert_array_equal(getattr(model, name), getattr(before, name))
    np.testing.assert_array_equal(
        model.geom("pot_bottom").friction, before.geom("pot_bottom").friction
    )
    assert info["maximum_radial_error_m"] < 0.000025
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    contacts = [
        c for c in data.contact[: data.ncon] if model.geom("pot_bottom").id in (c.geom1, c.geom2)
    ]
    assert contacts
    for contact in contacts:
        assert contact.dist > -1e-5
        normal = contact.frame[2] * (1 if contact.geom2 == model.geom("pot_bottom").id else -1)
        assert normal > 0.99
    assert model.geom("pot_bottom_visual").contype == 0
    assert model.geom("pot_bottom_visual").conaffinity == 0


def test_floor_conversion_rejects_implicit_mass_and_repeat_conversion():
    root, asset, pot = fixture()
    inertial = pot.find("inertial")
    pot.remove(inertial)
    with pytest.raises(ValueError, match="explicit pot inertia"):
        install_convex_pot_floor(asset, pot)
    pot.append(inertial)
    with pytest.raises(ValueError, match="segments"):
        install_convex_pot_floor(asset, pot, segments=63)
    install_convex_pot_floor(asset, pot)
    with pytest.raises(ValueError, match="original cylinder"):
        install_convex_pot_floor(asset, pot)
