import numpy as np
import pytest

from cap_harness.realman.supported_slide import supported_slide_settled
from cap_harness.realman.upper_handle import (
    OutletUnilateralAllowance,
    upper_bar_contacts,
    upper_handle_loads,
)


def test_outlet_allowance_keeps_other_failures_and_expires_at_clearance():
    gate = OutletUnilateralAllowance(0.2)
    evidence = dict(
        extracting=True,
        outlet_contact=True,
        platform_support_n=2,
        outlet_clear=False,
        unilateral_s=0.08,
        bilateral_s=0,
    )
    flags = [
        "persistent_unilateral_unload",
        "pad_asymmetric_overload",
        "pad_peak_load",
        "handle_translation",
        "persistent_bilateral_unload",
    ]
    remaining, allowed = gate.apply(flags, **evidence)
    assert remaining == flags[2:]
    assert allowed == flags[:2]
    assert gate.apply(flags, **{**evidence, "unilateral_s": 0.2}) == (flags, [])
    assert gate.apply(flags, **{**evidence, "outlet_clear": True}) == (flags, [])
    # Returning to the nozzle cannot re-arm an extraction-only allowance.
    assert gate.apply(flags, **evidence) == (flags, [])


@pytest.mark.parametrize(
    "changed",
    [
        {"extracting": False},
        {"outlet_contact": False},
        {"platform_support_n": 0.01},
        {"bilateral_s": 0.001},
    ],
)
def test_unilateral_allowance_requires_positive_support_and_one_remaining_contact(changed):
    evidence = dict(
        extracting=True,
        outlet_contact=True,
        platform_support_n=2,
        outlet_clear=False,
        unilateral_s=0.08,
        bilateral_s=0,
    )
    flags = ["persistent_unilateral_unload"]
    assert OutletUnilateralAllowance().apply(flags, **{**evidence, **changed}) == (flags, [])


def test_optional_heading_tolerance_preserves_legacy_default_and_other_limits():
    inputs = dict(
        position_error_m=0.001,
        yaw_error_deg=3,
        tilt_deg=1,
        linear_speed_m_s=0.001,
        angular_speed_rad_s=0.01,
        upward_support_n=2,
    )
    assert not supported_slide_settled(**inputs)
    assert supported_slide_settled(**inputs, max_yaw_error_deg=5)
    assert not supported_slide_settled(**{**inputs, "upward_support_n": 0}, max_yaw_error_deg=5)
    assert not supported_slide_settled(**{**inputs, "position_error_m": 0.003}, max_yaw_error_deg=5)
    with pytest.raises(ValueError):
        supported_slide_settled(**inputs, max_yaw_error_deg=float("nan"))


def test_upper_bar_bearing_requires_real_upward_contact_not_proximity():
    mj = pytest.importorskip("mujoco")
    m = mj.MjModel.from_xml_string("""<mujoco><option timestep=".001"/>
    <worldbody><site name="left_tool_mount"/>
    <body name="ltool_base_link"><site name="left_left_pad_surface" pos=".1 0 .11"/>
    <site name="left_right_pad_surface" pos=".09 0 .11"/>
    <body name="ltool_Right_Pad_Link"><geom name="left_right_pad_contact" type="box"
    pos=".095 0 .110" size=".005 .01 .008"/></body></body>
    <body name="pot"><freejoint name="pot_free"/><geom name="handle_collision_12"
    type="box" pos=".095 0 .125" size=".02 .015 .007" mass=".396"/></body>
    </worldbody></mujoco>""")
    d = mj.MjData(m)
    cfg = dict(
        handle_bottom_m=0.03,
        handle_outer_height_m=0.102,
        handle_outer_to_center_m=0.125,
        handle_outer_width_m=0.05,
        handle_rail_thickness_m=0.014,
    )
    d.qpos[2] = 0.01
    mj.mj_forward(m, d)
    assert upper_bar_contacts(m, d, cfg)["local_upward_normal_n"] == 0
    for _ in range(300):
        mj.mj_step(m, d)
    mj.mj_forward(m, d)
    evidence = upper_bar_contacts(m, d, cfg)
    assert evidence["local_upward_normal_n"] > 3
    assert all(c["normal_pot"][2] > 0.99 for c in evidence["contacts"])
    assert np.isfinite(evidence["side_approach_deviation_deg"])
    loads = upper_handle_loads(m, d, cfg)
    assert loads["finger_normal_loads_n"][1] > 3
    assert loads["finger_normal_resultants_n"][1] == pytest.approx(
        loads["finger_normal_loads_n"][1]
    )
    assert loads["finger_normal_vectors_world_n"][1][2] > 3
    assert not loads["opposed_contact"]  # Vertical support alone is not a grasp.
    assert max(loads["opposing_closing_forces_n"]) < 0.02


def test_handle_axis_uses_inward_extension_and_skips_ill_conditioned_seat_center():
    from cap_harness.realman.upper_handle import handle_axis_to_seat_error

    P = np.eye(4)
    P[0, 3] = 0.1
    assert handle_axis_to_seat_error(P, [0, 0, 0]) == pytest.approx(0)
    assert handle_axis_to_seat_error(P, [0.2, 0, 0]) == pytest.approx(0)
    assert handle_axis_to_seat_error(P, [0.1, 0.1, 0]) == pytest.approx(90)
    assert handle_axis_to_seat_error(P, [0.099, 0.001, 0]) is None


@pytest.mark.parametrize("height,bearing_expected", [(0.118, True), (0.05, False)])
def test_shallow_bearing_and_unqualified_finger_loads_are_not_hidden(height, bearing_expected):
    mj = pytest.importorskip("mujoco")
    normal = np.array([np.sin(np.deg2rad(65)), 0, np.cos(np.deg2rad(65))])
    center = np.array([0.095, 0, height])
    pot_center = center + normal * (0.008 + 0.007 - 0.0001)
    xyz = lambda p: " ".join(map(str, p))
    m = mj.MjModel.from_xml_string(f"""<mujoco><worldbody>
    <site name="left_tool_mount"/><body name="ltool_base_link">
    <site name="left_left_pad_surface" pos=".1 0 .11"/>
    <site name="left_right_pad_surface" pos=".09 0 .11"/>
    <body name="ltool_Right_Support_Link">
    <geom name="ltool_Right_Support_Link_collision_0" type="box"
      pos="{xyz(center)}" euler="0 65 0" size=".006 .01 .008"/>
    </body></body><body name="pot"><freejoint/>
    <geom name="handle_collision_3" type="box" pos="{xyz(pot_center)}"
      euler="0 65 0" size=".006 .01 .007" mass=".396"/>
    </body></worldbody></mujoco>""")
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    cfg = dict(
        handle_bottom_m=0.03,
        handle_outer_height_m=0.102,
        handle_outer_to_center_m=0.125,
        handle_outer_width_m=0.05,
        handle_rail_thickness_m=0.014,
    )
    evidence = upper_handle_loads(m, d, cfg)
    assert evidence["finger_normal_resultants_n"][1] > 0.02
    assert evidence["maximum_contact_normal_n"] > 0.02
    assert bool(evidence["contacts"]) is bearing_expected
    if bearing_expected:
        assert evidence["local_upward_normal_n"] > 0.02
        assert all(0 < c["normal_pot"][2] < 0.5 for c in evidence["contacts"])
        assert evidence["opposing_closing_forces_n"][1] > 0.02
    else:
        assert evidence["opposing_closing_forces_n"][1] == 0
    assert not evidence["opposed_contact"]
