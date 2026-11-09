"""Finite-patch spin resistance is an explicit scene candidate, not a pose lock."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cap_harness.realman.model import build_model

mj = pytest.importorskip("mujoco")


def test_opt_in_support_retains_translation_and_existing_friction(tmp_path):
    urdf = Path("ref/realman_robot/LSRX_V0.0.13.SLDASM/urdf/LSRX_V0.0.13.SLDASM.urdf")
    if not urdf.exists():
        pytest.skip("requires user-supplied RealBot meshes")
    cfg = json.loads(Path("configs/realman/coffee_scene.json").read_text())
    cfg["pot"]["support_contact_condim"] = 4
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    built = build_model(urdf, path, tmp_path / "model")
    tree = ET.parse(built["scene"])
    assert tree.find(".//geom[@name='pot_bottom']").get("condim") == "4"
    assert tree.find(".//body[@name='pot']/freejoint") is not None
    m = mj.MjModel.from_xml_path(str(built["scene"]))
    np.testing.assert_allclose(m.geom("pot_bottom").friction, [1, 0.005, 0.0001])
