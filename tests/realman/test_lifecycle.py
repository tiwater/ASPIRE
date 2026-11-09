import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cap_harness.realman.lifecycle import BODY_JOINTS, articulate_frozen_body


def test_observed_limit_conflict_is_explicit_and_work_fk_is_preserved(tmp_path):
    mj = pytest.importorskip("mujoco")
    angles = [-0.9, 0.6, 0.3, 0, 0]
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name="base_link")
    ET.SubElement(root, "actuator")
    urdf = ET.Element("robot", name="test")
    for name, angle in zip(BODY_JOINTS, angles):
        child = name.replace("joint", "link")
        quat = Rotation.from_rotvec([0, -angle, 0]).as_quat()[[3, 0, 1, 2]]
        body = ET.SubElement(body, "body", name=child, pos="0 0 .25", quat=" ".join(map(str, quat)))
        ET.SubElement(body, "inertial", pos="0 0 .1", mass="1", diaginertia=".1 .1 .1")
        joint = ET.SubElement(urdf, "joint", name=name, type="revolute")
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "axis", xyz="0 -1 0")
        ET.SubElement(
            joint, "limit", lower="-1.4", upper=".17" if name == "dt_joint" else "2", effort="120"
        )
    source = tmp_path / "frozen.xml"
    u = tmp_path / "robot.urdf"
    target = tmp_path / "dynamic.xml"
    ET.ElementTree(root).write(source)
    ET.ElementTree(urdf).write(u)
    cfg = {"robot": {"frozen_joints_rad": dict(zip(BODY_JOINTS, angles))}}
    with pytest.raises(ValueError, match="dt_joint"):
        articulate_frozen_body(source, cfg, u, target)
    assert not target.exists()
    meta = articulate_frozen_body(source, cfg, u, target, observed_envelope=True)
    assert not meta["hardware_limits_qualified"] and not meta["physical_motion_allowed"]
    assert meta["body_joints"][2]["legacy_range_rad"] == [-1.4, 0.17]
    sm = mj.MjModel.from_xml_path(str(source))
    sd = mj.MjData(sm)
    mj.mj_forward(sm, sd)
    m = mj.MjModel.from_xml_path(str(target))
    d = mj.MjData(m)
    for name, angle in zip(BODY_JOINTS, angles):
        d.joint(name).qpos[:] = angle
    mj.mj_forward(m, d)
    np.testing.assert_allclose(d.xpos, sd.xpos, atol=1e-12)
    np.testing.assert_allclose(d.xmat, sd.xmat, atol=1e-12)


def test_single_arm_exit_keeps_legal_hold_references_not_measured_soft_limit_error():
    from types import SimpleNamespace

    from cap_harness.realman.lifecycle import LifecycleMotion

    data = SimpleNamespace(qpos=np.zeros(26), ctrl=np.zeros(19))
    data.ctrl[2] = 1.0188534041442099
    data.qpos[2] = data.ctrl[2] + 3.6e-9
    data.ctrl[12:19] = np.linspace(-0.3, 0.3, 7)
    data.qpos[12:19] = data.ctrl[12:19] + 0.001
    data.qpos[19:] = np.arange(7)  # Uncontrolled object state is not changed.
    adapter = SimpleNamespace(
        data=data, qadr={"left": np.arange(5, 12), "right": np.arange(12, 19)}
    )
    motion = object.__new__(LifecycleMotion)
    motion.rollout = SimpleNamespace(adapter=adapter)
    motion.qids = motion.aids = np.arange(19)
    measured = data.qpos.copy()
    requested = np.linspace(-1, 1, 7)
    target = motion.arm_target("left", requested)
    assert target[2] == data.ctrl[2]
    np.testing.assert_array_equal(target[12:19], data.ctrl[12:19])
    np.testing.assert_array_equal(target[5:12], requested)
    np.testing.assert_array_equal(target[19:], data.qpos[19:])
    np.testing.assert_array_equal(data.qpos, measured)
    # Do not clip an invalid new arm target; the usual move validation must reject it.
    assert motion.arm_target("left", np.full(7, 100))[5] == 100
    with pytest.raises(ValueError):
        motion.arm_target("left", [0])
