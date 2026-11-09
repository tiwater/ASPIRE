"""The outlet spring's installed force must survive generation and compilation."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cap_harness.realman.model import build_model
from cap_harness.realman.outlet import install_conical_outlet

mj = pytest.importorskip("mujoco")


def hypothesis_config():
    config = json.loads(Path("configs/realman/coffee_scene.json").read_text())
    workload = json.loads(Path("configs/realman/coffee_workload_conical_outlet.json").read_text())
    config["outlet"]["conical_hypothesis"] = workload["outlet_conical_hypothesis"]
    return config


def test_conical_preload_is_preserved_in_generated_scene(tmp_path):
    urdf = Path("ref/realman_robot/LSRX_V0.0.13.SLDASM/urdf/LSRX_V0.0.13.SLDASM.urdf")
    if not urdf.exists():
        pytest.skip("requires user-supplied RealBot meshes")
    config = hypothesis_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    built = build_model(urdf, path, tmp_path / "model")
    m = mj.MjModel.from_xml_path(str(built["scene"]))
    j = m.joint("outlet_compression")
    assert m.jnt_stiffness[j.id] == 200
    np.testing.assert_allclose(m.jnt_solref[j.id], [0.002, 1])
    assert m.qpos0[j.qposadr[0]] == pytest.approx(0.003)
    assert m.qpos_spring[j.qposadr[0]] == pytest.approx(-0.002)
    np.testing.assert_allclose(m.geom("outlet_tip").friction, [0.3, 0.005, 0.0001])
    assert m.geom_priority[m.geom("outlet_tip").id] == 1
    stored = json.loads((tmp_path / "model/scene_parameters.json").read_text())
    assert stored["outlet_spring_trial"]["spring_force_seated_n"] == pytest.approx(0.8)
    assert stored["outlet_spring_trial"]["spring_force_at_cam_crest_n"] == pytest.approx(1.4)


@pytest.mark.parametrize(
    "field,value",
    [("height_m", 0.02), ("sliding_friction", float("nan")), ("bottom_radius_m", 0.008)],
)
def test_reject_invalid_or_unreachable_hypothesis(field, value):
    cfg = hypothesis_config()
    cfg["outlet"]["conical_hypothesis"][field] = value
    asset = ET.Element("asset")
    body = ET.Element("body", pos="0 0 0")
    with pytest.raises(ValueError):
        install_conical_outlet(asset, body, cfg, lid_initial_z_m=0.152)
    assert len(asset) == 0 and len(body) == 0
