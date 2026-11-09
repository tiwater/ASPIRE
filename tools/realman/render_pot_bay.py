"""Inspect actual pot-bay geometry with the head hidden for a top cutaway.

MUJOCO_GL=egl python tools/realman/render_pot_bay.py --model assets/realman/coffee/scene.xml
Only in-memory render groups change; the saved scene and collision model are untouched.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("assets/realman/coffee/scene.xml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/realman/pot-bay-review"))
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    groups = model.geom_group.copy()
    pot = model.body("pot").id
    selected = np.zeros(model.ngeom, dtype=bool)
    for i in range(model.ngeom):
        name = model.geom(i).name
        selected[i] = (
            model.geom_bodyid[i] == pot
            or name in ("machine_platform", "machine_back", "hotplate", "drip_tray")
            or name.startswith(("pot_bay_wall_collision_", "tray_slot"))
        )
    model.geom_group[~selected] = 5
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    args.output.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, width=960, height=720) as renderer:
        for name, elevation, show_pot in [
            ("pot_bay_top", -90, True),
            ("pot_bay_empty", -90, False),
            ("pot_bay_oblique", -35, False),
        ]:
            model.geom_group[model.geom_bodyid == pot] = (
                groups[model.geom_bodyid == pot] if show_pot else 5
            )
            camera = mujoco.MjvCamera()
            camera.lookat[:] = data.body("pot").xpos + [0.025, -0.045, 0.055]
            camera.distance = 0.65 if elevation == -90 else 0.55
            camera.orthographic = elevation == -90
            camera.azimuth = 0  # front at bottom of top views
            camera.elevation = elevation
            renderer.update_scene(data, camera=camera, scene_option=options)
            Image.fromarray(renderer.render()).save(args.output / f"{name}.png")


if __name__ == "__main__":
    main()
