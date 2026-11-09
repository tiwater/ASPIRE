"""The upright constraint applies to the composed body, including lateral lean."""

import numpy as np
import pytest

from cap_harness.realman.posture import upper_body_posture

mujoco = pytest.importorskip("mujoco")


def fixture(first=0, second=0, hip=0, lateral=0, yaw=0):
    return mujoco.MjModel.from_xml_string(
        f"""<mujoco><compiler angle="degree"/>
        <worldbody><body euler="0 {first} 0"><body pos="0 0 .255" euler="0 {second} 0">
        <body pos="0 0 .255" euler="0 {hip} 0"><body euler="{lateral} 0 0">
        <body name="xb_link" euler="0 0 {yaw}"/></body></body></body></body></worldbody>
        </mujoco>"""
    )


def test_folded_lower_joints_can_keep_upper_body_upright():
    report = upper_body_posture(fixture(80, -24, -58), maximum_tilt_deg=10)
    assert report["tilt_deg"] == pytest.approx(2)


def test_composed_tilt_rejects_small_individual_angles():
    with pytest.raises(ValueError, match="exceeds"):
        upper_body_posture(fixture(6, 6), maximum_tilt_deg=10)


def test_lateral_lean_is_counted_and_yaw_is_not_lean():
    assert upper_body_posture(fixture(yaw=90), maximum_tilt_deg=10)["tilt_deg"] == 0
    with pytest.raises(ValueError, match="exceeds"):
        upper_body_posture(fixture(first=8, lateral=8), maximum_tilt_deg=10)


@pytest.mark.parametrize("limit", [True, -1, 91, np.inf, np.nan])
def test_rejects_invalid_limit(limit):
    with pytest.raises(ValueError):
        upper_body_posture(fixture(), maximum_tilt_deg=limit)


def test_ten_degree_boundary_is_accepted():
    assert upper_body_posture(fixture(first=10), maximum_tilt_deg=10)["tilt_deg"] == pytest.approx(
        10
    )
