"""Render the actual scene's pot in isolation for geometry review (no simulation IO).

MUJOCO_GL=egl python tools/realman/render_pot.py --model assets/realman/coffee/scene.xml
"""

import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from PIL import Image


def isolated_pot(scene):
    source = ET.parse(scene).getroot()
    root = ET.Element("mujoco", model="coffee_pot_geometry_review")
    ET.SubElement(root, "compiler", angle="radian")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.3 0.3 0.3")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -0.4 0.6", dir="0 0 -1", diffuse="0.5 0.5 0.5")
    ET.SubElement(world, "geom", type="plane", size="1 1 0.01", rgba="0.24 0.27 0.30 1")
    pot = copy.deepcopy(source.find(".//body[@name='pot']"))
    pot.set("pos", "0 0 0")
    pot.attrib.pop("euler", None)
    pot.attrib.pop("quat", None)
    for item in list(pot):
        if item.tag in ("freejoint", "joint", "site"):
            pot.remove(item)
    world.append(pot)
    meshes = {g.get("mesh") for g in pot.findall("geom") if g.get("mesh")}
    for mesh in source.findall("asset/mesh"):
        if mesh.get("name") in meshes:
            asset.append(copy.deepcopy(mesh))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("assets/realman/coffee/scene.xml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/realman/pot-review"))
    args = parser.parse_args()
    model = isolated_pot(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    args.output.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, width=960, height=720) as renderer:
        for name, azimuth, elevation in [
            ("side", 90, -4),
            ("oblique", 135, -24),
            ("rear", 160, -16),
            ("top", 90, -89),
            ("spout", 45, -20),
        ]:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.025, 0, 0.075]
            camera.distance, camera.azimuth, camera.elevation = 0.43, azimuth, elevation
            renderer.update_scene(data, camera=camera)
            Image.fromarray(renderer.render()).save(args.output / f"pot_{name}.png")


if __name__ == "__main__":
    main()
