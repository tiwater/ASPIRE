import copy
import json
from pathlib import Path

import numpy as np
import pytest

from cap_harness.realman.capture import image_array
from cap_harness.realman.telemetry import (
    commissioning_report,
    decode_state,
    millidegrees_to_radians,
)


def sample():
    raw = {}
    for key, n in [("arm_left", 7), ("arm_right", 7), ("body", 5), ("head", 2)]:
        raw[key] = {"position": [0] * n, "joint_err": [0] * n, "joint_en": [1] * n, "sys_err": 0}
    raw["arm_left"]["position"][0] = 90000
    for arm in ["left", "right"]:
        raw["hand_" + arm] = {"position": [200]}
    return raw


def test_native_units_and_no_fabricated_fk():
    state = decode_state(sample(), received_at_s=10)
    assert state.joints_rad["left"][0] == pytest.approx(np.pi / 2)
    assert state.gripper_native["left"] == 200
    assert state.as_dict()["end_effector_poses"] is None
    assert not state.as_dict()["urdf_mapping_qualified"]


@pytest.mark.parametrize("bad", [[0] * 6, [float("nan")] * 7, [True] * 7, ["0"] * 7])
def test_reject_bad_joint_data(bad):
    with pytest.raises(ValueError):
        millidegrees_to_radians(bad)


def test_errors_are_not_motion_readiness():
    raw = sample()
    raw["arm_right"]["joint_err"][2] = 4
    assert decode_state(raw, received_at_s=1).faults
    config = json.loads(Path("configs/realman/robot_sim.json").read_text())
    report = commissioning_report(config)
    assert "controller_to_urdf_left" in report["pending"]
    assert not report["physical_motion_available"]
    # Calibration records cannot certify the separately gated live runtime.
    config["commissioning"] = {"test": {"status": "verified", "evidence_path": "somewhere"}}
    assert not commissioning_report(config)["physical_motion_available"]


def test_image_padding_and_big_endian(tmp_path):
    (tmp_path / "d.bin").write_bytes(bytes([0, 1, 1, 0, 9, 9, 0, 2, 2, 0, 9, 9]))
    m = {
        "encoding": "16UC1",
        "is_bigendian": True,
        "height": 2,
        "width": 2,
        "step": 6,
        "file": "d.bin",
    }
    np.testing.assert_array_equal(image_array(tmp_path, m), [[1, 256], [2, 512]])
    m["step"] = 8
    with pytest.raises(ValueError):
        image_array(tmp_path, m)


def test_faulty_gripper_not_clipped_to_valid():
    raw = copy.deepcopy(sample())
    raw["hand_left"]["position"] = [1001]
    with pytest.raises(ValueError):
        decode_state(raw, received_at_s=1)
