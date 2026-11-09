"""Independent gripper tuning must not change geometry, friction or the other hand."""

import xml.etree.ElementTree as ET

import pytest

from cap_harness.realman.gripper_tuning import (
    GripperSimulationProfile,
    apply_gripper_profile,
    resize_pad_plane,
)


def fixture_xml():
    root = ET.fromstring("""<mujoco><option timestep=".001"/><worldbody><body>
      <joint name="ltool_Left_1_Joint" actuatorfrcrange="-10 10"/>
      <geom name="left_left_pad_contact" size=".001 .018 .009" friction=".8 .005 .0001" solref=".005 1"/>
      <geom name="left_right_pad_contact" size=".001 .018 .009" friction=".8 .005 .0001" solref=".005 1"/>
      <geom name="right_left_pad_contact" size=".001 .018 .009" solref=".005 1"/>
      </body></worldbody><actuator>
      <position name="left_gripper_servo" joint="ltool_Left_1_Joint" kp="36" kv=".2" forcerange="-3 3"/>
      <position name="right_gripper_servo" kp="4" kv=".2" forcerange="-1 1"/>
      </actuator><equality/></mujoco>""")
    for name in ["Left_Support", "Left_2", "Right_2", "Right_1", "Right_Support"]:
        ET.SubElement(root.find(".//body"), "joint", name=f"ltool_{name}_Joint", damping=".03")
        ET.SubElement(
            root.find("equality"),
            "joint",
            joint1=f"ltool_{name}_Joint",
            joint2="ltool_Left_1_Joint",
            solref=".004 1",
        )
    return root


def test_force_and_stiffness_are_opt_in_and_do_not_change_friction_or_geometry():
    root = fixture_xml()
    other = ET.tostring(root.find(".//position[@name='right_gripper_servo']"))
    pads = [node.attrib.copy() for node in root.findall(".//geom")]
    result = apply_gripper_profile(
        root, "left", GripperSimulationProfile(120, 0.4, 6, 0.002, 0.004, 0.1)
    )
    assert result["simulation_only"] and not result["material_strength_modeled"]
    assert all(float(node.get("damping")) == 0.1 for node in root.findall(".//body/joint"))
    assert ET.tostring(root.find(".//position[@name='right_gripper_servo']")) == other
    for before, after in zip(pads, root.findall(".//geom")):
        for key in before.keys() - {"solref"}:
            assert after.get(key) == before[key]
        assert after.get("solref") == (
            "0.004 1" if before["name"].startswith("left_") else ".005 1"
        )
    assert root.find(".//position[@name='left_gripper_servo']").get("forcerange") == "-6 6"


@pytest.mark.parametrize(
    "profile",
    [
        GripperSimulationProfile(120, 0.4, 11),
        GripperSimulationProfile(float("nan"), 0.4, 6),
        GripperSimulationProfile(120, 0.4, 6, 0.001, 0.004),
    ],
)
def test_invalid_profile_is_rejected_before_any_mutation(profile):
    root = fixture_xml()
    before = ET.tostring(root)
    with pytest.raises(ValueError):
        apply_gripper_profile(root, "left", profile)
    assert ET.tostring(root) == before


def pad_fixture():
    root = ET.fromstring("<mujoco><worldbody/></mujoco>")
    for arm in ("left", "right"):
        for side in ("Left", "Right"):
            body = ET.SubElement(
                root.find("worldbody"), "body", name=f"{arm[0]}tool_{side}_Pad_Link"
            )
            ET.SubElement(
                body,
                "geom",
                name=f"{arm}_{side.lower()}_pad_contact",
                type="box",
                size=".001 .018 .009",
                mass=".002",
                friction=".8 .005 .0001",
            )
            ET.SubElement(body, "site", name=f"{arm}_{side.lower()}_pad_surface", pos="-.001 0 0")
    return root


def test_pad_plane_change_preserves_density_and_normal_surface():
    import numpy as np

    mj = pytest.importorskip("mujoco")
    from cap_harness.realman.payload import body_mass_properties

    root = pad_fixture()
    other = ET.tostring(root.find(".//body[@name='rtool_Left_Pad_Link']"))
    resize_pad_plane(root, "left", pad_y_span_m=0.036, pad_z_span_m=0.060)
    assert ET.tostring(root.find(".//body[@name='rtool_Left_Pad_Link']")) == other
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    mass = 0.002 * 60 / 18
    properties = body_mass_properties(model, "ltool_Left_Pad_Link")
    assert properties.mass_kg == pytest.approx(mass)
    expected = mass / 3 * np.diag([0.018**2 + 0.030**2, 0.001**2 + 0.030**2, 0.001**2 + 0.018**2])
    np.testing.assert_allclose(properties.inertia_kg_m2, expected, atol=1e-12)
    np.testing.assert_allclose(model.site("left_left_pad_surface").pos, [-0.001, 0, 0])
    np.testing.assert_allclose(model.geom("left_left_pad_contact").friction, [0.8, 0.005, 0.0001])


def test_explicit_pad_inertia_cannot_be_left_stale_by_resize():
    root = pad_fixture()
    ET.SubElement(root.find(".//body[@name='ltool_Right_Pad_Link']"), "inertial")
    original = ET.tostring(root)
    with pytest.raises(ValueError, match="geometry-derived inertia"):
        resize_pad_plane(root, "left", pad_y_span_m=0.036, pad_z_span_m=0.060)
    assert ET.tostring(root) == original
